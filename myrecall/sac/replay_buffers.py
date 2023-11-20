import random
from typing import Dict

import numpy as np
import tensorflow as tf


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
        )


class ReservoirReplayBuffer(ReplayBuffer):
    """Buffer for SAC agents implementing reservoir sampling."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        super().__init__(obs_dim, act_dim, size)
        self.timestep = 0

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        current_t = self.timestep
        self.timestep += 1

        if current_t < self.max_size:
            buffer_idx = current_t
        else:
            buffer_idx = random.randint(0, current_t)
            if buffer_idx >= self.max_size:
                return

        self.obs_buf[buffer_idx] = obs
        self.next_obs_buf[buffer_idx] = next_obs
        self.actions_buf[buffer_idx] = action
        self.rewards_buf[buffer_idx] = reward
        self.done_buf[buffer_idx] = done
        self.size = min(self.size + 1, self.max_size)


class ExpertReplayBuffer:
    """A expert experience replay buffer for behavioral cloning,
    which does not support overwriting old samples."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.policy_mu_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.policy_std_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.q1_buf = np.zeros([size,], dtype=np.float32)
        self.q2_buf = np.zeros([size,], dtype=np.float32)
        self.size, self.max_size = 0, size

    def store(self, obs: np.ndarray, actions: np.ndarray, policy_mu: np.ndarray, policy_std: np.ndarray,
              q1: np.ndarray, q2: np.ndarray) -> None:
        assert self.size + obs.shape[0] <= self.max_size
        range_start = self.size
        range_end = self.size + obs.shape[0]
        self.obs_buf[range_start:range_end] = obs
        self.actions_buf[range_start:range_end] = actions
        self.policy_mu_buf[range_start:range_end] = policy_mu
        self.policy_std_buf[range_start:range_end] = policy_std
        self.q1_buf[range_start:range_end] = q1
        self.q2_buf[range_start:range_end] = q2
        self.size = range_end

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            policy_mu=tf.convert_to_tensor(self.policy_mu_buf[idxs]),
            policy_std=tf.convert_to_tensor(self.policy_std_buf[idxs]),
            q1=tf.convert_to_tensor(self.q1_buf[idxs]),
            q2=tf.convert_to_tensor(self.q2_buf[idxs]),
        )


class PerfectReplayBuffer:
    """A simple Perfect replay buffer for SAC agents."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, steps_per_task: int) -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.steps_per_task, self.size, self.max_size = steps_per_task, 0, size

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        assert self.size < self.max_size, "Out of perfect memory!"
        self.obs_buf[self.size] = obs
        self.next_obs_buf[self.size] = next_obs
        self.actions_buf[self.size] = action
        self.rewards_buf[self.size] = reward
        self.done_buf[self.size] = done
        self.size = self.size + 1

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        cur_task_t = self.size % self.steps_per_task
        cur_task_idx = self.size // self.steps_per_task

        idxs = np.random.randint(self.size - cur_task_t, self.size, size=batch_size)
        if cur_task_idx > 0:
            idxs = np.append(idxs, np.random.randint(0, self.size - cur_task_t, size=batch_size))

        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
        )

    # def sample_cur_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
    #     cur_task_t = self.size % self.steps_per_task
    #     idxs = np.random.randint(self.size - cur_task_t, self.size, size=batch_size)
    #
    #     return dict(
    #         obs=tf.convert_to_tensor(self.obs_buf[idxs]),
    #         next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
    #         actions=tf.convert_to_tensor(self.actions_buf[idxs]),
    #         rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
    #         done=tf.convert_to_tensor(self.done_buf[idxs]),
    #     )
    #
    # def sample_his_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
    #     cur_task_t = self.size % self.steps_per_task
    #     idxs = np.random.randint(0, self.size - cur_task_t, size=batch_size)
    #
    #     return dict(
    #         obs=tf.convert_to_tensor(self.obs_buf[idxs]),
    #         next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
    #         actions=tf.convert_to_tensor(self.actions_buf[idxs]),
    #         rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
    #         done=tf.convert_to_tensor(self.done_buf[idxs]),
    #     )

    def sample_cur_batch_for_bc(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(self.size - self.steps_per_task, self.size, size=batch_size)

        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
        )


# class PerfectReplayBuffer_:
#     """A simple Perfect replay buffer for SAC agents."""
#
#     def __init__(self, obs_dim: int, act_dim: int, size: int, steps_per_task: int) -> None:
#         self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
#         self.rewards_buf = np.zeros(size, dtype=np.float32)
#         self.done_buf = np.zeros(size, dtype=np.float32)
#
#         self.policy_mu_buf = np.zeros([size, act_dim], dtype=np.float32)
#         self.policy_std_buf = np.zeros([size, act_dim], dtype=np.float32)
#         self.q1_buf = np.zeros([size,], dtype=np.float32)
#         self.q2_buf = np.zeros([size,], dtype=np.float32)
#
#         self.steps_per_task, self.size, self.max_size = steps_per_task, 0, size
#
#     def store(
#         self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
#     ) -> None:
#         assert self.size < self.max_size, "Out of perfect memory!"
#         self.obs_buf[self.size] = obs
#         self.next_obs_buf[self.size] = next_obs
#         self.actions_buf[self.size] = action
#         self.rewards_buf[self.size] = reward
#         self.done_buf[self.size] = done
#         self.size = self.size + 1
#
#     def store_target(self, policy_mu: np.ndarray, policy_std: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> None:
#         self.policy_mu_buf = policy_mu
#         self.policy_std_buf = policy_std
#         self.q1_buf = q1
#         self.q2_buf = q2
#
#     def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
#         cur_task_t = self.size % self.steps_per_task
#         cur_task_idx = self.size // self.steps_per_task
#
#         idxs = np.random.randint(self.size - cur_task_t, self.size, size=batch_size)
#         if cur_task_idx > 0:
#             idxs = np.append(idxs, np.random.randint(0, self.size - cur_task_t, size=batch_size))
#
#         return dict(
#             obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#             next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
#             actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#             rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
#             done=tf.convert_to_tensor(self.done_buf[idxs]),
#         )
#
#     # def sample_cur_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
#     #     cur_task_t = self.size % self.steps_per_task
#     #     idxs = np.random.randint(self.size - cur_task_t, self.size, size=batch_size)
#     #
#     #     return dict(
#     #         obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#     #         next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
#     #         actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#     #         rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
#     #         done=tf.convert_to_tensor(self.done_buf[idxs]),
#     #     )
#     #
#     # def sample_his_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
#     #     cur_task_t = self.size % self.steps_per_task
#     #     idxs = np.random.randint(0, self.size - cur_task_t, size=batch_size)
#     #
#     #     return dict(
#     #         obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#     #         next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
#     #         actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#     #         rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
#     #         done=tf.convert_to_tensor(self.done_buf[idxs]),
#     #     )
#
#     def sample_cur_batch_for_target(self, batch_size: int) -> Dict[str, tf.Tensor]:
#         idxs = np.arange(self.size - self.steps_per_task, self.size)
#
#         return dict(
#             obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#             next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
#             actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#             rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
#             done=tf.convert_to_tensor(self.done_buf[idxs]),
#         )
#
#     def sample_batch_for_bc(self, batch_size: int) -> Dict[str, tf.Tensor]:
#         cur_task_t = self.size % self.steps_per_task
#         idxs = np.random.randint(0, self.size - cur_task_t, size=batch_size)
#
#         return dict(
#             obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#             actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#             policy_mu=tf.convert_to_tensor(self.policy_mu_buf[idxs]),
#             policy_std=tf.convert_to_tensor(self.policy_std_buf[idxs]),
#             q1=tf.convert_to_tensor(self.q1_buf[idxs]),
#             q2=tf.convert_to_tensor(self.q2_buf[idxs]),
#         )
