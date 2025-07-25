#!/bin/bash

# Training script for 0% masking (baseline)
echo "Starting training with 0% masking (baseline)..."

python -u train.py \
    -n "pong-mask-0p-50k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_0.yaml" \
    -env_name "ALE/Pong-v5" \
    -trajectory_path "D_TRAJ/Pong.pkl" \

echo "Training with 0% masking completed!" 