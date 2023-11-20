import argparse

from continualworld.tasks import TASK_SEQS
from continualworld.utils.enums import BufferType
from continualworld.utils.utils import float_or_str, sci2int, str2bool


def cl_parse_args(args=None):
    parser = argparse.ArgumentParser(description="Continual World")

    parser.add_argument(
        "--tasks",
        type=str,
        choices=TASK_SEQS.keys(),
        default='CW20',
        help="Name of the sequence you want to run",
    )
    parser.add_argument(
        "--logger_output",
        type=str,
        nargs="+",
        choices=["neptune", "tensorboard", "tsv"],
        default=["tensorboard", "tsv"],
        help="Types of logger used.",
    )
    parser.add_argument("--seed", type=int, help="Seed for randomness")
    parser.add_argument(
        "--steps_per_task", type=sci2int, default=int(1e6), help="Numer of steps per task"
    )
    parser.add_argument(
        "--log_every",
        type=sci2int,
        default=int(2e4),
        help="Number of steps between subsequent evaluations and logging",
    )
    parser.add_argument(
        "--replay_size", type=sci2int, default=int(1e6), help="Size of the replay buffer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Minibatch size for the optimization"
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 256, 256, 256],
        # default=[5, 5, 5],
        help="Hidden sizes list for the MLP models",
    )
    parser.add_argument(
        "--buffer_type",
        type=str,
        default="fifo",
        choices=[b.value for b in BufferType],
        help="Strategy of inserting examples into the buffer",
    )
    parser.add_argument(
        "--reset_buffer_on_task_change",
        type=str2bool,
        default=True,
        help="If true, replay buffer is reset on each task change",
    )
    parser.add_argument(
        "--reset_optimizer_on_task_change",
        type=str2bool,
        default=True,
        help="If true, optimizer is reset on each task change",
    )
    parser.add_argument(
        "--reset_critic_on_task_change",
        type=str2bool,
        default=False,
        help="If true, critic model is reset on each task change",
    )
    parser.add_argument(
        "--activation", type=str, default="lrelu", help="Activation kind for the models"
    )
    parser.add_argument(
        "--use_layer_norm",
        type=str2bool,
        default=True,
        help="Whether or not use layer normalization",
    )
    parser.add_argument(
        "--use_separate_critic",
        type=str2bool,
        default=False,
        help="Whether or not use separate critic network for each task",
    )
    parser.add_argument(
        "--use_multi_layer_head",
        type=str2bool,
        default=False,
        help="Whether or not use multi-layer head",
    )
    parser.add_argument(
        "--use_popArt",
        type=str2bool,
        default=False,
        help="Whether or not use popArt for critic",
    )
    parser.add_argument(
        "--carried_critic",
        type=str2bool,
        default=False,
        help="Whether or not carry the learned critic",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--alpha",
        default="auto",
        help="Entropy regularization coefficient. "
        "Can be either float value, or 'auto', in which case it is dynamically tuned.",
    )
    parser.add_argument(
        "--target_output_std",
        type=float,
        default=0.089,
        help="If alpha is 'auto', alpha is dynamically tuned so that standard deviation "
        "of the action distribution on every dimension matches target_output_std.",
    )
    parser.add_argument(
        "--cl_method",
        type=str,
        choices=["ft", "ewc", "clonex", "packnet", "pm", "recall"],
        default=None,
        help="If None, finetuning method will be used. If one of 'l2', 'ewc', 'mas', 'vcl',"
        "'packnet', 'agem', respective method will be used.",
    )
    parser.add_argument(
        "--packnet_retrain_steps",
        type=int,
        default=0,
        help="Number of retrain steps after network pruning, which occurs after each task",
    )
    parser.add_argument(
        "--regularize_critic",
        type=str2bool,
        default=False,
        help="If True, both actor and critic are regularized; if False, only actor is",
    )
    parser.add_argument(
        "--behavior_cloning",
        type=str2bool,
        default=False,
        help="If True, use behavior cloning for both actor and critic",
    )
    parser.add_argument(
        "--cl_reg_coef",
        type=float,
        default=0.0,
        help="Regularization strength for continual learning methods. "
        "Valid for 'ewc' continual learning methods.",
    )
    parser.add_argument(
        "--policy_reg_coef",
        type=float,
        default=0.0,
        help="Regularization strength for continual learning methods. "
             "Valid for 'clonex', 'recall' continual learning methods.",
    )
    parser.add_argument(
        "--value_reg_coef",
        type=float,
        default=0.0,
        help="Regularization strength for continual learning methods. "
             "Valid for 'clonex', 'recall' continual learning methods.",
    )
    parser.add_argument(
        "--multihead_archs", type=str2bool, default=True, help="Whether use multi-head architecture"
    )
    parser.add_argument(
        "--hide_task_id",
        type=str2bool,
        default=True,
        help="if True, one-hot encoding of the task will not be appended to observation",
    )
    parser.add_argument("--clipnorm", type=float, default=None, help="Value for gradient clipping")
    parser.add_argument(
        "--agent_policy_exploration",
        type=str2bool,
        default=False,
        help="If True, uniform exploration for start_steps steps is used only in the first task"
        "(in continual learning). Otherwise, it is used in every task",
    )
    return parser.parse_args(args=args)
