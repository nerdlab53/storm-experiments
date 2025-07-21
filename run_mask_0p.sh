#!/bin/bash

# Training script for 0% masking (baseline)
echo "Starting training with 0% masking (baseline)..."

python -u train.py \
    -n "pong-mask-0p-20k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_25.yaml" \
    -env_name "ALE/Pong-v5" \
    -trajectory_path "D_TRAJ/Pong.pkl" \
    --override "Models.WorldModel.FixedMaskPercent=0.0"

echo "Training with 0% masking completed!" 