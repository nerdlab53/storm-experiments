#!/bin/bash

# Training script for 5% masking
echo "Starting training with 10% masking..."

python -u train.py \
    -n "pong-mask-10p-10k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_10.yaml" \
    -env_name "ALE/Pong-v5" \
    -trajectory_path "D_TRAJ/Pong.pkl" \

echo "Training with 10% masking completed!" 