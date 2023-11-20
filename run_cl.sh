#!/usr/bin/env bash

# baselines
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 --cl_method=ft

CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=ewc --cl_reg_coef=10000.0

CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=packnet --packnet_retrain_steps=100000 --clipnorm=2e-05

CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=pm --batch_size=512 --buffer_type=reservoir --reset_buffer_on_task_change=False --replay_size=6e6

CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=clonex --policy_reg_coef=100.0 --agent_policy_exploration=True --clipnorm=0.1

# proposed method
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=recall --batch_size=128 --replay_size=6e6 \
  --behavior_cloning=True --policy_reg_coef=10.0 \
  --use_multi_layer_head=True --use_popArt=True --agent_policy_exploration=True --carried_critic=True


# pairwise tasks: pm
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW0_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=pm --batch_size=512 --buffer_type=reservoir --reset_buffer_on_task_change=False --replay_size=2e6

# pairwise tasks: recall
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW0_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=recall --batch_size=128 --replay_size=2e6 \
  --behavior_cloning=True --policy_reg_coef=0.01 \
  --use_popArt=True

# ablations: pm, pm+popart, pm+bc, recall
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=recall --batch_size=128 --replay_size=6e6 \
  --behavior_cloning=False --policy_reg_coef=10.0 \
  --use_multi_layer_head=True --use_popArt=False --agent_policy_exploration=True --carried_critic=True
