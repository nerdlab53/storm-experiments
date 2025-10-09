#!/bin/bash
# Experiment 1: Baseline STORM (no adaptive masking)
# Encoder: 8×8 = 64 spatial tokens (with 128 channels)
# No spatial token masking applied

env_name=Breakout
python -u train.py \
    -n "${env_name}-baseline-8x8-wm_2L512D8H-50k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM_Baseline_8x8.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl"

