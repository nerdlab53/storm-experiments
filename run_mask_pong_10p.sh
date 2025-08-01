#!/bin/bash

# Training script for 0% masking (baseline)
echo "Starting training with 10% masking (baseline)..."

python -u train.py \
    -n "pong-mask-10p-100k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_10.yaml" \
    -env_name "ALE/Pong-v5" \   
    -trajectory_path "D_TRAJ/Pong.pkl" \

echo "Training with 10% masking completed!" 