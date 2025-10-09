#!/bin/bash
# Experiment 3: AdaMAE with Adaptive Sampler + MaskNet
# Encoder: 8×8 = 64 spatial tokens (with 128 channels)
# Learns WHICH tokens to mask via REINFORCE (sampler)
# Learns HOW MANY tokens to mask via REINFORCE (MaskNet, unconstrained)
# Both trained jointly with reconstruction loss as reward

env_name=Breakout
python -u train.py \
    -n "${env_name}-adamae-full-8x8-learnedratio-wm_2L512D8H-50k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM_AdaMAE_Full_8x8.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    --use_adamae

