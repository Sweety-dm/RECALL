#!/usr/bin/env bash

# for plot_intro(args)
python3 produce_results.py --cl_logs logs/intro_pm --baseline_logs logs/single --output_path experiments/intro_pm
#python3 produce_results.py --cl_logs logs/intro_recall --baseline_logs logs/single --output_path experiments/intro_recall


# for CW36_metrics(args)
python3 produce_results.py --cl_logs logs/results --baseline_logs logs/single --output_path experiments/results


# for plot_ablations(args)
python3 produce_results.py --cl_logs logs/ablations --baseline_logs logs/single --output_path experiments/ablations


# for main(args)
for task in 'CW10' 'CW20'
do
python3 produce_results.py --task $task --cl_logs logs/results/$task --baseline_logs logs/single --output_path experiments/results
done