from typing import Callable, Iterable, List, Tuple

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model

from myrecall.envs import MW_ACT_LEN, MW_OBS_LEN

EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(x: tf.Tensor, mu: tf.Tensor, log_std: tf.Tensor) -> tf.Tensor:
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(input_tensor=pre_sum, axis=1)


def apply_squashing_func(
    mu: tf.Tensor, pi: tf.Tensor, logp_pi
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Adjustment to log prob
    # NOTE: This formula is a little bit magic. To get an understanding of where it
    # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
    # appendix C. This is a more numerically-stable equivalent to Eq 21.
    # Try deriving it yourself as a (very difficult) exercise. :)
    logp_pi -= tf.reduce_sum(input_tensor=2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


def mlp(
    input_dim: int, hidden_sizes: Iterable[int], activation: Callable, use_layer_norm: bool = False
) -> Model:
    model = tf.keras.Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(hidden_sizes[0]))
    if use_layer_norm:
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Activation(tf.nn.tanh))
    else:
        model.add(tf.keras.layers.Activation(activation))
    for size in hidden_sizes[1:]:
        model.add(tf.keras.layers.Dense(size, activation=activation))
    return model

def mlp_head(
    input_dim: int, hidden_sizes: Iterable[int], activation: Callable, num_heads: int
) -> Model:
    x = [0] * num_heads
    inputs = tf.keras.Input(shape=(input_dim,))
    for i in range(num_heads):
        x[i] = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)(inputs)
        for size in hidden_sizes[1:]:
            x[i] = tf.keras.layers.Dense(size, activation=activation)(x[i])
        x[i] = tf.keras.layers.Dense(1)(x[i])
    head_out = tf.concat(x, axis=-1)

    model = tf.keras.Model(inputs=inputs, outputs=head_out)

    return model

def separate_network(
    input_dim: int, hidden_sizes: Iterable[int], activation: Callable, num_heads: int, use_layer_norm: bool = False
) -> Model:
    x = [0] * num_heads
    inputs = tf.keras.Input(shape=(input_dim,))
    for i in range(num_heads):
        x[i] = tf.keras.layers.Dense(hidden_sizes[0])(inputs)
        if use_layer_norm:
            x[i] = tf.keras.layers.LayerNormalization()(x[i])
            x[i] = tf.keras.layers.Activation(tf.nn.tanh)(x[i])
        else:
            x[i] = tf.keras.layers.Activation(activation)(x[i])
        for size in hidden_sizes[1:]:
            x[i] = tf.keras.layers.Dense(size, activation=activation)(x[i])
        x[i] = tf.keras.layers.Dense(1)(x[i])
    head_out = tf.concat(x, axis=-1)

    model = tf.keras.Model(inputs=inputs, outputs=head_out)

    return model

# def separate_network(
#     input_dim: int, hidden_sizes: Iterable[int], activation: Callable, num_heads: int, use_layer_norm: bool = False
# ) -> Model:
#     models = [None] * num_heads
#     for i in range(num_heads):
#         models[i] = tf.keras.Sequential()
#         models[i].add(Input(shape=(input_dim,)))
#         models[i].add(tf.keras.layers.Dense(hidden_sizes[0]))
#         if use_layer_norm:
#             models[i].add(tf.keras.layers.LayerNormalization())
#             models[i].add(tf.keras.layers.Activation(tf.nn.tanh))
#         else:
#             models[i].add(tf.keras.layers.Activation(activation))
#         for size in hidden_sizes[1:]:
#             models[i].add(tf.keras.layers.Dense(size, activation=activation))
#         models[i].add(tf.keras.layers.Dense(1))
#     return models

def _choose_head(out: tf.Tensor, obs: tf.Tensor, num_heads: int) -> tf.Tensor:
    """For multi-head output, choose appropriate head.

    We assume that task number is one-hot encoded as a part of observation.

    Args:
      out: multi-head output tensor from the model
      obs: obsevation batch. We assume that last num_heads dims is one-hot encoding of task
      num_heads: number of heads

    Returns:
      tf.Tensor: output for the appropriate head
    """
    batch_size = tf.shape(out)[0]
    out = tf.reshape(out, [batch_size, -1, num_heads])
    obs = tf.reshape(obs[:, -num_heads:], [batch_size, num_heads, 1])
    return tf.squeeze(out @ obs, axis=2)


class MlpActor(Model):
    def __init__(
        self,
        input_dim: int,
        action_space: gym.Space,
        hidden_sizes: Iterable[int] = (256, 256),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super(MlpActor, self).__init__()
        self.num_heads = num_heads
        # if True, one-hot encoding of the task will not be appended to observation.
        self.hide_task_id = hide_task_id

        if self.hide_task_id:
            input_dim = MW_OBS_LEN

        self.core = mlp(input_dim, hidden_sizes, activation, use_layer_norm=use_layer_norm)
        self.head_mu = tf.keras.Sequential(
            [
                Input(shape=(hidden_sizes[-1],)),
                tf.keras.layers.Dense(action_space.shape[0] * num_heads),
            ]
        )
        self.head_log_std = tf.keras.Sequential(
            [
                Input(shape=(hidden_sizes[-1],)),
                tf.keras.layers.Dense(action_space.shape[0] * num_heads),
            ]
        )
        self.action_space = action_space

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        obs = x
        if self.hide_task_id:
            x = x[:, :MW_OBS_LEN]
        x = self.core(x)
        mu = self.head_mu(x)
        log_std = self.head_log_std(x)

        if self.num_heads > 1:
            mu = _choose_head(mu, obs, self.num_heads)
            log_std = _choose_head(log_std, obs, self.num_heads)

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(input=mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        # Make sure actions are in correct range
        action_scale = self.action_space.high[0]
        mu *= action_scale
        pi *= action_scale

        return mu, log_std, pi, logp_pi

    def predict_pi(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        obs = x
        if self.hide_task_id:
            x = x[:, :MW_OBS_LEN]
        x = self.core(x)
        mu = self.head_mu(x)
        log_std = self.head_log_std(x)

        if self.num_heads > 1:
            mu = _choose_head(mu, obs, self.num_heads)
            log_std = _choose_head(log_std, obs, self.num_heads)

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        return mu, std

    @property
    def common_variables(self) -> List[tf.Variable]:
        """Get model parameters which are shared for each task. This excludes head parameters
        in the multi-head setting, as they are separate for each task."""
        if self.num_heads > 1:
            return self.core.trainable_variables
        elif self.num_heads == 1:
            return (
                self.core.trainable_variables
                + self.head_mu.trainable_variables
                + self.head_log_std.trainable_variables
            )


class MlpCritic(Model):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int] = (256, 256),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        num_heads: int = 1,
        use_separate_critic: bool = False,
        use_multi_layer_head: bool = False,
        hide_task_id: bool = False,
    ) -> None:
        super(MlpCritic, self).__init__()
        self.hide_task_id = hide_task_id
        self.num_heads = (
            num_heads  # if True, one-hot encoding of the task will not be appended to observation.
        )
        self.use_separate_critic = use_separate_critic
        self.use_multi_layer_head = use_multi_layer_head

        if self.hide_task_id:
            input_dim = MW_OBS_LEN + MW_ACT_LEN

        if self.use_separate_critic:
            self.core = tf.keras.Sequential([Input(shape=(input_dim,))])
            self.head = separate_network(input_dim, hidden_sizes, activation, num_heads, use_layer_norm=use_layer_norm)
        elif self.use_multi_layer_head:
            self.core = mlp(input_dim, hidden_sizes[:1], activation, use_layer_norm=use_layer_norm)
            self.head = mlp_head(hidden_sizes[0], hidden_sizes[1:], activation, num_heads)
        else:
            self.core = mlp(input_dim, hidden_sizes, activation, use_layer_norm=use_layer_norm)
            self.head = tf.keras.Sequential(
                [Input(shape=(hidden_sizes[-1],)), tf.keras.layers.Dense(num_heads)]
            )

    def call(self, x: tf.Tensor, a: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        obs = x
        if self.hide_task_id:
            x = x[:, :MW_OBS_LEN]
        x = self.head(self.core(tf.concat([x, a], axis=-1)))
        if self.num_heads > 1:
            x = _choose_head(x, obs, self.num_heads)
        x = tf.squeeze(x, axis=1)
        return x

    @property
    def common_variables(self) -> List[tf.Variable]:
        """Get model parameters which are shared for each task. This excludes head parameters
        in the multi-head setting, as they are separate for each task."""
        if self.num_heads > 1:
            return self.core.trainable_variables
        elif self.num_heads == 1:
            return self.core.trainable_variables + self.head.trainable_variables


class PopArtMlpCritic(MlpCritic):
    """PopArt implementation.

    PopArt is a method for normalizing returns, especially useful in multi-task learning.
    See https://arxiv.org/abs/1602.07714 and https://arxiv.org/abs/1809.04474v1.
    """

    def __init__(self, beta=3e-4, **kwargs) -> None:
        super(PopArtMlpCritic, self).__init__(**kwargs)

        self.moment1 = tf.Variable(tf.zeros((self.num_heads, 1)), trainable=False)
        self.moment2 = tf.Variable(tf.ones((self.num_heads, 1)), trainable=False)
        self.sigma = tf.Variable(tf.ones((self.num_heads, 1)), trainable=False)

        self.beta = beta
        # print(self.head.layers)
        # print(self.head.trainable_variables)

    @tf.function
    def unnormalize(self, x: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
        moment1 = tf.squeeze(obs[:, -self.num_heads :] @ self.moment1, axis=1)
        sigma = tf.squeeze(obs[:, -self.num_heads :] @ self.sigma, axis=1)
        return x * sigma + moment1

        # return tf.sign(x) * (tf.exp(tf.abs(x)) - 1)

    @tf.function
    def normalize(self, x: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
        moment1 = tf.squeeze(obs[:, -self.num_heads :] @ self.moment1, axis=1)
        sigma = tf.squeeze(obs[:, -self.num_heads :] @ self.sigma, axis=1)
        return (x - moment1) / sigma

        # return tf.sign(x) * tf.math.log(tf.abs(x) + 1)

    @tf.function
    def update_stats(self, returns: tf.Tensor, obs: tf.Tensor) -> None:
        task_counts = tf.reduce_sum(obs[:, -self.num_heads :], axis=0)
        batch_moment1 = tf.reduce_sum(
            tf.expand_dims(returns, 1) * obs[:, -self.num_heads :], axis=0
        ) / tf.math.maximum(task_counts, 1.0)
        batch_moment2 = tf.reduce_sum(
            tf.expand_dims(returns * returns, 1) * obs[:, -self.num_heads :], axis=0
        ) / tf.math.maximum(task_counts, 1.0)

        update_pos = tf.expand_dims(tf.cast(task_counts > 0, tf.float32), 1)
        new_moment1 = self.moment1 + update_pos * (
            self.beta * (tf.expand_dims(batch_moment1, 1) - self.moment1)
        )
        new_moment2 = self.moment2 + update_pos * (
            self.beta * (tf.expand_dims(batch_moment2, 1) - self.moment2)
        )
        new_sigma = tf.math.sqrt(new_moment2 - new_moment1 * new_moment1)
        new_sigma = tf.clip_by_value(new_sigma, 1e-4, 1e6)

        # Update weights of the last layer.
        if self.use_multi_layer_head or self.use_separate_critic:
            for i in range(self.num_heads):
                last_layer = self.head.layers[-1 - self.num_heads + i]
                last_layer.kernel.assign(
                    last_layer.kernel * tf.transpose(self.sigma[i]) / tf.transpose(new_sigma[i])
                )
                last_layer.bias.assign(
                    (last_layer.bias * tf.squeeze(self.sigma[i]) + tf.squeeze(self.moment1[i] - new_moment1[i]))
                    / tf.squeeze(new_sigma[i])
                )
                # print(last_layer.kernel)
        else:
            last_layer = self.head.layers[-1]
            last_layer.kernel.assign(
                last_layer.kernel * tf.transpose(self.sigma) / tf.transpose(new_sigma)
            )
            last_layer.bias.assign(
                (last_layer.bias * tf.squeeze(self.sigma) + tf.squeeze(self.moment1 - new_moment1))
                / tf.squeeze(new_sigma)
            )

        self.moment1.assign(new_moment1)
        self.moment2.assign(new_moment2)
        self.sigma.assign(new_sigma)