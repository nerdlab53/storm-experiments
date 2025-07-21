#!/bin/bash

# Training script for 15% masking
echo "Starting training with 15% masking..."

python -u train.py \
    -n "pong-mask-15p-20k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_25.yaml" \
    -env_name "ALE/Pong-v5" \
    -trajectory_path "D_TRAJ/Pong.pkl" \
    --override "Models.WorldModel.FixedMaskPercent=0.15"

echo "Training with 15% masking completed!" 