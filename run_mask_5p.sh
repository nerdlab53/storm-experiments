#!/bin/bash

# Training script for 5% masking
echo "Starting training with 5% masking..."

python -u train.py \
    -n "pong-mask-5p-20k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_25.yaml" \
    -env_name "ALE/Pong-v5" \
    -trajectory_path "D_TRAJ/Pong.pkl" \
    --override "Models.WorldModel.FixedMaskPercent=0.05"

echo "Training with 5% masking completed!" 