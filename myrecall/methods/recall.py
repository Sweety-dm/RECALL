import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from myrecall.sac.replay_buffers import PerfectReplayBuffer, ExpertReplayBuffer
from myrecall.sac.sac import SAC
from myrecall.sac import models

from myrecall.utils.utils import reset_optimizer
from myrecall.envs import MW_OBS_LEN


class RECALL_SAC(SAC):
    def __init__(self, behavior_cloning=False, policy_reg_coef=1.0, value_reg_coef=1.0, regularize_critic=False,
                 carried_critic=False, use_multi_layer_head=False, use_separate_critic=False,
                 **vanilla_sac_kwargs):
        """Perfect Memory Plus method.
        """
        super().__init__(**vanilla_sac_kwargs)

        self.replay_buffer = PerfectReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size, steps_per_task=self.env.steps_per_env
        )

        self.policy_reg_coef = policy_reg_coef
        self.value_reg_coef = value_reg_coef
        self.behavior_cloning = behavior_cloning
        self.regularize_critic = regularize_critic
        self.carried_critic = carried_critic
        self.use_multi_layer_head = use_multi_layer_head
        self.use_separate_critic = use_separate_critic

        if self.behavior_cloning:
            self.clipnorm = 0.1

            self.memory_size_per_task = 10000
            self.expert_buffer = ExpertReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                    size=int(self.num_tasks * self.memory_size_per_task))


    def on_task_end(self, current_task_idx: int) -> None:
        if self.behavior_cloning:
            batch = self.replay_buffer.sample_cur_batch_for_bc(self.memory_size_per_task)
            obs, actions = batch["obs"], batch["actions"]
            policy_mu, policy_std = self.actor.predict_pi(obs)
            policy_mu = tf.stop_gradient(policy_mu)
            policy_std = tf.stop_gradient(policy_std)

            q1 = tf.stop_gradient(self.critic1(obs, actions))
            q2 = tf.stop_gradient(self.critic2(obs, actions))

            self.expert_buffer.store(obs, actions, policy_mu, policy_std, q1, q2)
        else:
            pass

    def get_auxiliary_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        if self.behavior_cloning and current_task_idx > 0:
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

    def get_gradients(
        self,
        seq_idx: tf.Tensor,
        aux_batch: Dict[str, tf.Tensor],
        obs: tf.Tensor,
        next_obs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        done: tf.Tensor,
    ) -> Tuple[Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]], Dict]:
        with tf.GradientTape(persistent=True) as g:
            if self.auto_alpha:
                log_alpha = self.get_log_alpha(obs)
            else:
                log_alpha = tf.math.log(self.alpha)

            # Main outputs from computation graph
            mu, log_std, pi, logp_pi = self.actor(obs)
            q1 = self.critic1(obs, actions)
            q2 = self.critic2(obs, actions)

            # compose q with pi, for pi-learning
            q1_pi = self.critic1(obs, pi)
            q2_pi = self.critic2(obs, pi)

            # get actions and log probs of actions for next states, for Q-learning
            _, _, pi_next, logp_pi_next = self.actor(next_obs)

            # target q values, using actions from *current* policy
            target_q1 = self.target_critic1(next_obs, pi_next)
            target_q2 = self.target_critic2(next_obs, pi_next)

            # Min Double-Q:
            min_q_pi = tf.minimum(q1_pi, q2_pi)
            min_target_q = tf.minimum(target_q1, target_q2)

            # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
            if self.critic_cl is models.PopArtMlpCritic:
                q_backup = tf.stop_gradient(
                    self.critic1.normalize(
                        rewards
                        + self.gamma
                        * (1 - done)
                        * (
                                self.critic1.unnormalize(min_target_q, next_obs)
                                - tf.math.exp(log_alpha) * logp_pi_next
                        ),
                        obs,
                    )
                )
            else:
                q_backup = tf.stop_gradient(
                    rewards
                    + self.gamma
                    * (1 - done)
                    * (min_target_q - tf.math.exp(log_alpha) * logp_pi_next)
                )

            # Soft actor-critic losses
            pi_loss = tf.reduce_mean(
                tf.math.exp(log_alpha[:self.batch_size]) * logp_pi[:self.batch_size] - min_q_pi[:self.batch_size])
            q1_loss = 0.5 * tf.reduce_mean((q_backup[:self.batch_size] - q1[:self.batch_size]) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup[:self.batch_size] - q2[:self.batch_size]) ** 2)
            if tf.shape(obs)[0] > self.batch_size:
                pi_loss += tf.multiply(tf.cast(seq_idx, dtype=tf.float32), tf.reduce_mean(
                    tf.math.exp(log_alpha[self.batch_size:]) * logp_pi[self.batch_size:] - min_q_pi[self.batch_size:]))
                q1_loss += tf.multiply(tf.cast(seq_idx, dtype=tf.float32), 0.5 * tf.reduce_mean(
                    (q_backup[self.batch_size:] - q1[self.batch_size:]) ** 2))
                q2_loss += tf.multiply(tf.cast(seq_idx, dtype=tf.float32), 0.5 * tf.reduce_mean(
                    (q_backup[self.batch_size:] - q2[self.batch_size:]) ** 2))
            value_loss = q1_loss + q2_loss

            auxiliary_loss = self.get_auxiliary_loss(seq_idx, aux_batch)
            metrics = dict(
                pi_loss=pi_loss,
                q1_loss=q1_loss,
                q2_loss=q2_loss,
                q1=q1,
                q2=q2,
                logp_pi=logp_pi,
                reg_loss=auxiliary_loss,
            )

            pi_loss += auxiliary_loss
            value_loss += auxiliary_loss

            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    log_alpha[:self.batch_size] * tf.stop_gradient(
                        logp_pi[:self.batch_size] + self.target_entropy
                    ))
                # if tf.shape(obs)[0] > self.batch_size:
                #     alpha_loss += tf.multiply(tf.cast(seq_idx, dtype=tf.float32), -tf.reduce_mean(
                #         log_alpha[self.batch_size:] * tf.stop_gradient(
                #             logp_pi[self.batch_size:] + self.target_entropy
                #         )))

        # Compute gradients
        actor_gradients = g.gradient(pi_loss, self.actor.trainable_variables)
        critic_gradients = g.gradient(value_loss, self.critic_variables)
        if self.auto_alpha:
            alpha_gradient = g.gradient(alpha_loss, self.all_log_alpha)
        else:
            alpha_gradient = None
        del g

        if self.critic_cl is models.PopArtMlpCritic:
            self.critic1.update_stats(q_backup, obs)

        gradients = (actor_gradients, critic_gradients, alpha_gradient)
        return gradients, metrics

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

        if (self.agent_policy_exploration and current_task_idx > 0):
            best_return_head = self.get_best_return_head(self.num_test_eps_stochastic,
                                                         current_task_idx,
                                                         self.num_tasks)

            # Initialize the weights of actor head corresponding to the current task to that of the best-return head
            start, end = self.act_dim * current_task_idx, self.act_dim * (current_task_idx + 1)
            best_start, best_end = self.act_dim * best_return_head, self.act_dim * (best_return_head + 1)

            head_mu_weights = self.actor.head_mu.get_weights()
            head_log_std_weights = self.actor.head_log_std.get_weights()

            head_mu_weights[0][:, start:end] = head_mu_weights[0][:, best_start:best_end]
            head_mu_weights[1][start:end] = head_mu_weights[1][best_start:best_end]
            head_log_std_weights[0][:, start:end] = head_log_std_weights[0][:, best_start:best_end]
            head_log_std_weights[1][start:end] = head_log_std_weights[1][best_start:best_end]

            self.actor.head_mu.set_weights(head_mu_weights)
            self.actor.head_log_std.set_weights(head_log_std_weights)

            # alpha for actor network
            # if self.auto_alpha:
            #     self.all_log_alpha[current_task_idx].assign(self.all_log_alpha[best_return_head])

            # Initialize the weights of actor head corresponding to the current task to that of the best-return head

            if (self.carried_critic and current_task_idx > 0):
                if self.use_multi_layer_head or self.use_separate_critic:
                    start_, end_ = 2 * current_task_idx, 2 * (current_task_idx + 1)
                    best_start_, best_end_ = 2 * best_return_head, 2 * (best_return_head + 1)

                    critic1_head_weights = self.critic1.head.get_weights()
                    critic2_head_weights = self.critic2.head.get_weights()

                    critic1_head_weights = np.reshape(critic1_head_weights, (-1, 2 * self.num_tasks))
                    critic1_head_weights[:, start_:end_] = critic1_head_weights[:, best_start_:best_end_]
                    critic1_head_weights = np.reshape(critic1_head_weights, (1, -1))[0]
                    critic2_head_weights = np.reshape(critic2_head_weights, (-1, 2 * self.num_tasks))
                    critic2_head_weights[:, start_:end_] = critic2_head_weights[:, best_start_:best_end_]
                    critic2_head_weights = np.reshape(critic2_head_weights, (1, -1))[0]

                    self.critic1.head.set_weights(critic1_head_weights)
                    self.critic2.head.set_weights(critic2_head_weights)

                else:
                    critic1_head_weights = self.critic1.head.get_weights()
                    critic2_head_weights = self.critic2.head.get_weights()

                    critic1_head_weights[0][:, current_task_idx] = critic1_head_weights[0][:, best_return_head]
                    critic1_head_weights[1][current_task_idx] = critic1_head_weights[1][best_return_head]
                    critic2_head_weights[0][:, current_task_idx] = critic2_head_weights[0][:, best_return_head]
                    critic2_head_weights[1][current_task_idx] = critic2_head_weights[1][best_return_head]

                    self.critic1.head.set_weights(critic1_head_weights)
                    self.critic2.head.set_weights(critic2_head_weights)

                # print(self.critic1.head.trainable_variables)
                # print(self.critic2.head.trainable_variables)
                self.target_critic1.set_weights(self.critic1.get_weights())
                self.target_critic2.set_weights(self.critic2.get_weights())

                # popArt paras for critic
                # if self.critic_cl is models.PopArtMlpCritic:
                #     self.critic1.moment1[current_task_idx].assign(self.critic1.moment1[best_return_head])
                #     self.critic1.moment2[current_task_idx].assign(self.critic1.moment2[best_return_head])
                #     self.critic1.sigma[current_task_idx].assign(self.critic1.sigma[best_return_head])

        if self.reset_optimizer_on_task_change:
            reset_optimizer(self.optimizer)

        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)
        self.all_common_variables = (
                self.actor.common_variables
                + self.critic1.common_variables
                + self.critic2.common_variables
        )
