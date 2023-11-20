import argparse
import os

import pandas as pd
import numpy as np

from myrecall.results_processing.plots import plot_intro_metrics, plot_intro_plasticity_badcase, \
    plot_intro_forgetting_case, plot_average_performance
from myrecall.results_processing.tables import calculate_metrics, calculate_intro_metrics
from myrecall.results_processing.utils import METHODS_ORDER, get_data_for_runs
from myrecall.utils.utils import str2bool


def main(args: argparse.Namespace) -> None:
    # mtl_data = get_data_for_runs(args.mtl_logs, kind="mtl")
    baseline_data = get_data_for_runs(args.baseline_logs, kind="single")
    cl_data = get_data_for_runs(args.cl_logs, kind="cl")

    output_dir = os.path.join(args.output_path, f"{args.tasks}")
    os.makedirs(output_dir, exist_ok=True)

    table = calculate_metrics(cl_data, baseline_data, methods_order=METHODS_ORDER)
    table = table.round(2)
    table_path = os.path.join(output_dir, "results.csv")
    table.to_csv(table_path)

    print(f"Report saved to {output_dir}")


def plot_intro(args: argparse.Namespace) -> None:
    ### plot performance matrix in introduction
    results = []
    for i in range(10):
        for j in range(10):
            path = os.path.join(args.cl_logs, str("CW{}_{}").format(i, j))
            cl_data = get_data_for_runs(path, kind="cl")
            result = calculate_intro_metrics(cl_data)

            result.index = [str("CW{}_{}").format(i, j)]
            result.index.name = "tasks"
            results.append(result)
    results = pd.concat(results)

    os.makedirs(args.output_path, exist_ok=True)
    table_path = os.path.join(args.output_path, "metrics.csv")
    results.to_csv(table_path)
    plot_intro_metrics(results, args.output_path)
    print(str('Average T1 performance:{:.2f}\n').format(np.mean(results.values[:, 0])),
          str('Average T2 performance:{:.2f}\n').format(np.mean(results.values[:, 3])),
          str('Average T1 forgetting:{:.2f}\n').format(np.mean(results.values[:, 6])),)

    ### plot current task learning curve in a bad plasticity case for example
    path = os.path.join(args.cl_logs, "CW5_0")
    data = get_data_for_runs(path, kind="cl")
    plot_intro_plasticity_badcase(data, args.output_path)

    tasks = ['CW1_3', 'CW1_4', 'CW2_2', 'CW2_3', 'CW2_8',
             'CW3_6', 'CW4_3', 'CW7_1', 'CW7_9', 'CW8_0',
             'CW8_2', 'CW8_4', 'CW8_6']
    task_labels = ['(1,3)', '(1,4)', '(2,2)', '(2,3)', '(2,8)',
                   '(3,6)', '(4,3)', '(7,1)', '(7,9)', '(8,0)',
                   '(8,2)', '(8,4)', '(8,6)']
    plot_intro_forgetting_case(args.cl_logs, args.output_path, tasks, task_labels)

    print(f"Report saved to {args.output_path}")



def CW36_metrics(args: argparse.Namespace) -> None:
    baseline_data = get_data_for_runs(args.baseline_logs, kind="single")
    CW3 = ['CW3_0', 'CW3_1', 'CW3_2', 'CW3_3', 'CW3_4', 'CW3_5', 'CW3_6', 'CW3_7']
    CW6 = ['CW6_0', 'CW6_1', 'CW6_2', 'CW6_3', 'CW6_4', 'CW6_5', 'CW6_6', 'CW6_7']

    for tasks in ['CW3', 'CW6']:
        output_dir = os.path.join(args.output_path, f"{tasks}")
        os.makedirs(output_dir, exist_ok=True)
        performance, forgetting, transfer = [], [], []
        if tasks == 'CW3': tasks = CW3
        if tasks == 'CW6': tasks = CW6
        for task in tasks:
            cl_data = get_data_for_runs(os.path.join(args.cl_logs, task), kind="cl")
            metrics = calculate_metrics(cl_data, baseline_data, methods_order=METHODS_ORDER)

            for metric_name in ['performance', 'forgetting', 'total_normalized_ft']:
                columns = []
                for method in [m for m in METHODS_ORDER if m in metrics.index]:
                    columns.extend([rf"{method}_{metric_name}", rf"{method}_lb_{metric_name}", rf"{method}_ub_{metric_name}"])
                metric = metrics.loc[:,metric_name:rf'ub_{metric_name}'].values.reshape((1,-1))[0]
                result = pd.DataFrame([np.around(metric, 2)], columns=columns)
                result.insert(0, 'task', task)

                if metric_name == 'performance':
                    performance.append(result)
                elif metric_name == 'forgetting':
                    forgetting.append(result)
                elif metric_name == 'total_normalized_ft':
                    transfer.append(result)

        performance = pd.concat(performance)
        forgetting = pd.concat(forgetting)
        transfer = pd.concat(transfer)

        performance.to_csv(os.path.join(output_dir, "performance.csv"))
        forgetting.to_csv(os.path.join(output_dir, "forgetting.csv"))
        transfer.to_csv(os.path.join(output_dir, "transfer.csv"))

        print(f"Report saved to {output_dir}")


def plot_ablations(args: argparse.Namespace) -> None:
    CW6 = ['CW6_0', 'CW6_1', 'CW6_2', 'CW6_3', 'CW6_4', 'CW6_5', 'CW6_6', 'CW6_7']

    for task in CW6:
        path = os.path.join(args.cl_logs, task)
        data = get_data_for_runs(path, kind="cl")

        output_path = os.path.join(args.output_path, task)
        os.makedirs(output_path, exist_ok=True)
        plot_average_performance(data, output_path)

    print(f"Report saved to {args.output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str)
    parser.add_argument("--cl_logs", type=str)
    parser.add_argument("--baseline_logs", type=str)
    parser.add_argument(
        "--use_ci",
        type=str2bool,
        default=False,
        help="When True, confidence intervals are shown for every plot. Note that plots may be significantly "
             "slower to generate."
    )
    parser.add_argument("--visualize", type=str2bool, default="False")
    parser.add_argument("--output_path", type=str, default="experiments")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_intro(args)
    CW36_metrics(args)
    plot_ablations(args)
    main(args)