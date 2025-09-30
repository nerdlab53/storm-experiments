#!/bin/bash
# Train AdaMAE with fixed 50% masking
env_name=Breakout
python -u train.py \
    -n "${env_name}-life_done-AdaMAE-50p-50k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM_AdaMAE_50p.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    --use_adamae
