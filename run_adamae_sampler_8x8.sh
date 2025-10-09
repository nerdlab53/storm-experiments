#!/bin/bash
# Experiment 2: AdaMAE with Adaptive Sampler ONLY (no MaskNet)
# Encoder: 8×8 = 64 spatial tokens (with 128 channels)
# Learns WHICH tokens to mask via REINFORCE
# Uses FIXED mask ratio: 50% (32 visible tokens, MAX allowed)
# MaskNet disabled

env_name=Breakout
python -u train.py \
    -n "${env_name}-adamae-sampler-8x8-50p-wm_2L512D8H-50k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM_AdaMAE_Sampler_8x8.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    --use_adamae

