#!/bin/bash

# Training script for 25% masking (higher sparsity test)
echo "Starting training with 25% masking..."

python -u train.py \
    -n "pong-mask-25p-50k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_25.yaml" \
    -env_name "ALE/Pong-v5" \
    -trajectory_path "D_TRAJ/Pong.pkl" \

echo "Training with 25% masking completed!" 