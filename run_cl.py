import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

from typing import Callable, Iterable

from myrecall.envs import get_cl_env, get_single_env
from myrecall.sac.utils.logx import EpochLogger
from myrecall.sac import models
from myrecall.tasks import TASK_SEQS
from myrecall.utils.enums import BufferType
from myrecall.utils.run_utils import get_sac_class
from myrecall.utils.utils import get_activation_from_str
from input_args import cl_parse_args


def main(
    logger: EpochLogger,
    tasks: str,
    seed: int,
    steps_per_task: int,
    log_every: int,
    replay_size: int,
    batch_size: int,
    hidden_sizes: Iterable[int],
    buffer_type: str,
    reset_buffer_on_task_change: bool,
    reset_optimizer_on_task_change: bool,
    activation: Callable,
    use_layer_norm: bool,
    use_separate_critic: bool,
    use_multi_layer_head: bool,
    carried_critic: bool,
    use_popArt: bool,
    lr: float,
    gamma: float,
    alpha: str,
    target_output_std: float,
    cl_method: str,
    packnet_retrain_steps: int,
    regularize_critic: bool,
    cl_reg_coef: float,
    policy_reg_coef:float,
    value_reg_coef:float,
    behavior_cloning: bool,
    reset_critic_on_task_change: bool,
    multihead_archs: bool,
    hide_task_id: bool,
    clipnorm: float,
    agent_policy_exploration: bool,
):
    tasks_list = TASK_SEQS[tasks]
    print(tasks_list)
    train_env = get_cl_env(tasks_list, steps_per_task)
    # Consider normalizing test envs in the future.
    num_tasks = len(tasks_list)
    test_envs = [
        get_single_env(task, one_hot_idx=i, one_hot_len=num_tasks) for i, task in enumerate(tasks_list)
    ]
    steps = steps_per_task * len(tasks_list)

    num_heads = num_tasks if multihead_archs else 1
    actor_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        use_layer_norm=use_layer_norm,
        num_heads=num_heads,
        hide_task_id=hide_task_id,
    )
    critic_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        use_layer_norm=use_layer_norm,
        use_separate_critic=use_separate_critic,
        use_multi_layer_head=use_multi_layer_head,
        num_heads=num_heads,
        hide_task_id=hide_task_id,
    )

    if use_popArt:
        critic_cl = models.PopArtMlpCritic
    else:
        critic_cl = models.MlpCritic

    vanilla_sac_kwargs = {
        "tasks": tasks,
        "method": cl_method,
        "env": train_env,
        "test_envs": test_envs,
        "logger": logger,
        "seed": seed,
        "steps": steps,
        "log_every": log_every,
        "replay_size": replay_size,
        "batch_size": batch_size,
        "critic_cl": critic_cl,
        "actor_kwargs": actor_kwargs,
        "critic_kwargs": critic_kwargs,
        "buffer_type": BufferType(buffer_type),
        "reset_buffer_on_task_change": reset_buffer_on_task_change,
        "reset_optimizer_on_task_change": reset_optimizer_on_task_change,
        "lr": lr,
        "alpha": alpha,
        "reset_critic_on_task_change": reset_critic_on_task_change,
        "clipnorm": clipnorm,
        "gamma": gamma,
        "target_output_std": target_output_std,
        "agent_policy_exploration": agent_policy_exploration,
    }

    sac_class = get_sac_class(cl_method)

    if cl_method in ["ft", "pm"]:
        sac = sac_class(**vanilla_sac_kwargs)
    elif cl_method in ["ewc"]:
        sac = sac_class(
            **vanilla_sac_kwargs, cl_reg_coef=cl_reg_coef, regularize_critic=regularize_critic
        )
    elif cl_method in ["clonex"]:
        sac = sac_class(
            **vanilla_sac_kwargs,
            policy_reg_coef=policy_reg_coef, value_reg_coef=value_reg_coef, regularize_critic=regularize_critic
        )
    elif cl_method in ["recall"]:
        sac = sac_class(
            **vanilla_sac_kwargs,
            behavior_cloning=behavior_cloning,
            policy_reg_coef=policy_reg_coef,
            value_reg_coef=value_reg_coef,
            regularize_critic=regularize_critic,
            carried_critic=carried_critic,
            use_multi_layer_head=use_multi_layer_head,
            use_separate_critic=use_separate_critic
        )
    elif cl_method == "packnet":
        sac = sac_class(
            **vanilla_sac_kwargs,
            regularize_critic=regularize_critic,
            retrain_steps=packnet_retrain_steps
        )
    else:
        raise NotImplementedError("This method is not implemented")
    sac.run()


if __name__ == "__main__":
    args = vars(cl_parse_args())
    logger = EpochLogger(args["logger_output"], config=args, tasks=args["tasks"],
                         method=args["cl_method"], seed=args["seed"])
    del args["logger_output"]
    main(logger, **args)
