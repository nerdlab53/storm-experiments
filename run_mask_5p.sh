#!/bin/bash

# Training script for 5% masking
echo "Starting training with 5% masking..."

python -u train.py \
    -n "pong-mask-5p-50k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_5.yaml" \
    -env_name "ALE/Pong-v5" \
    -trajectory_path "D_TRAJ/Pong.pkl" \

echo "Training with 5% masking completed!" 