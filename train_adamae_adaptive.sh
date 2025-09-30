#!/bin/bash
# Train AdaMAE with adaptive masking schedule (25% → 75% over 20k steps)
env_name=Breakout
python -u train.py \
    -n "${env_name}-life_done-AdaMAE-adaptive-50k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM_AdaMAE_adaptive.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    --use_adamae
