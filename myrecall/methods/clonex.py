from typing import Dict, Optional

import tensorflow as tf
import numpy as np

from myrecall.sac.replay_buffers import ReplayBuffer, ExpertReplayBuffer
from myrecall.sac.sac import SAC

from myrecall.utils.enums import BufferType
from myrecall.utils.utils import reset_optimizer, reset_weights

from myrecall.envs import MW_OBS_LEN


class ClonEx_SAC(SAC):
    def __init__(self, policy_reg_coef=1.0, value_reg_coef=1.0, regularize_critic=False, **vanilla_sac_kwargs):
        """Class for behavioral cloning methods.
        """
        super().__init__(**vanilla_sac_kwargs)
        self.policy_reg_coef = policy_reg_coef
        self.value_reg_coef = value_reg_coef
        self.regularize_critic = regularize_critic
        self.memory_size_per_task = 10000
        self.expert_buffer = ExpertReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                size=int(self.num_tasks * self.memory_size_per_task))

    def on_task_end(self, current_task_idx: int) -> None:
        batch = self.replay_buffer.sample_batch(self.memory_size_per_task)
        obs, actions = batch["obs"], batch["actions"]
        policy_mu, policy_std = self.actor.predict_pi(obs)
        policy_mu = tf.stop_gradient(policy_mu)
        policy_std = tf.stop_gradient(policy_std)

        q1 = tf.stop_gradient(self.critic1(obs, actions))
        q2 = tf.stop_gradient(self.critic2(obs, actions))

        self.expert_buffer.store(obs, actions, policy_mu, policy_std, q1, q2)

    def get_auxiliary_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        if current_task_idx > 0:
            expert_batch = self.expert_buffer.sample_batch(self.batch_size)
        else:
            expert_batch = None
        return expert_batch

    def get_auxiliary_loss(self, current_task_idx: tf.Tensor, aux_batch: Dict[str, tf.Tensor],) -> tf.Tensor:
        if aux_batch is None:
            aux_loss = tf.constant(0.0)
        else:
            aux_loss = self.auxiliary_loss(**aux_batch)
        return aux_loss

    def auxiliary_loss(self, obs: tf.Tensor, actions: tf.Tensor, policy_mu: tf.Tensor, policy_std: tf.Tensor,
                       q1: tf.Tensor, q2: tf.Tensor) -> tf.Tensor:
        cur_policy_mu, cur_policy_std = self.actor.predict_pi(obs)
        expert_policy = tf.compat.v1.distributions.Normal(loc=policy_mu, scale=policy_std)
        cur_policy = tf.compat.v1.distributions.Normal(loc=cur_policy_mu, scale=cur_policy_std)
        aux_loss = tf.constant(self.policy_reg_coef) * \
                   tf.reduce_mean(tf.compat.v1.distributions.kl_divergence(cur_policy, expert_policy))
        if self.regularize_critic:
            cur_q1 = self.critic1(obs, actions)
            cur_q2 = self.critic2(obs, actions)
            critic_loss = 0.5 * tf.reduce_mean((cur_q1 - q1) ** 2) + 0.5 * tf.reduce_mean((cur_q2 - q2) ** 2)
            aux_loss += tf.constant(self.value_reg_coef) * critic_loss

        return aux_loss

    def get_best_return_head(self, num_episodes, current_task_idx, num_heads) -> int:
        ave_returns = []
        test_env = self.test_envs[current_task_idx]
        one_hot_encoder = np.zeros(num_heads)
        for head_idx in range(current_task_idx):
            one_hot = one_hot_encoder
            one_hot[head_idx] = 1.0
            episode_returns = []
            for j in range(num_episodes):
                obs, done, episode_return, episode_len = test_env.reset(), False, 0, 0
                while not (done or (episode_len == self.max_episode_len)):
                    obs[MW_OBS_LEN:] = one_hot
                    obs, reward, done, _ = test_env.step(
                        self.get_action_test(tf.convert_to_tensor(obs))
                    )
                    episode_return += reward
                    episode_len += 1
                episode_returns.append(episode_return)
            ave_returns.append(np.mean(episode_returns))

        return ave_returns.index(max(ave_returns))

    def _handle_task_change(self, current_task_idx: int):
        self.on_task_start(current_task_idx)

        if self.reset_buffer_on_task_change:
            assert self.buffer_type == BufferType.FIFO
            self.replay_buffer = ReplayBuffer(
                obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size
            )
        if self.reset_critic_on_task_change:
            reset_weights(self.critic1, self.critic_cl, self.critic_kwargs)
            self.target_critic1.set_weights(self.critic1.get_weights())
            reset_weights(self.critic2, self.critic_cl, self.critic_kwargs)
            self.target_critic2.set_weights(self.critic2.get_weights())

        if (self.agent_policy_exploration and current_task_idx > 0):
            best_return_head = self.get_best_return_head(self.num_test_eps_stochastic,
                                                         current_task_idx,
                                                         self.num_tasks)
            # Initialize the weights of actor head corresponding to the current task to that of the best-return head
            start, end = self.act_dim * current_task_idx, self.act_dim * (current_task_idx + 1)
            start_, end_ = self.act_dim * best_return_head, self.act_dim * (best_return_head + 1)

            head_mu_weights = self.actor.head_mu.get_weights()
            head_log_std_weights = self.actor.head_log_std.get_weights()

            head_mu_weights[0][:, start:end] = head_mu_weights[0][:, start_:end_]
            head_mu_weights[1][start:end] = head_mu_weights[1][start_:end_]
            head_log_std_weights[0][:, start:end] = head_log_std_weights[0][:, start_:end_]
            head_log_std_weights[1][start:end] = head_log_std_weights[1][start_:end_]

            self.actor.head_mu.set_weights(head_mu_weights)
            self.actor.head_log_std.set_weights(head_log_std_weights)

        if self.reset_optimizer_on_task_change:
            reset_optimizer(self.optimizer)

        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)
        self.all_common_variables = (
                self.actor.common_variables
                + self.critic1.common_variables
                + self.critic2.common_variables
        )
