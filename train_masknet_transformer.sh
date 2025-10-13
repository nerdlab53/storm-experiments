#!/bin/bash

# Training script for improved MaskNet with Transformer architecture
# This script tests the new MaskNet architecture that matches the AdaMAE sampler
# with residual connections and layer normalization for better stability

env_name=Breakout
experiment_name="${env_name}-AdaMAE-MaskNetTransformer-4x4-50k-seed1"

echo "=========================================="
echo "Training with Improved MaskNet (Transformer)"
echo "=========================================="
echo "Environment: ${env_name}"
echo "Experiment: ${experiment_name}"
echo "Config: STORM_AdaMAE_MaskNetTransformer_4x4.yaml"
echo "Features:"
echo "  - MaskNet Type: transformer (with residuals + LayerNorm)"
echo "  - Encoder: 4x4 grid (16 tokens per frame)"
echo "  - Loss coefficients: L_S_which=0.1, L_S_ratio=0.1"
echo "  - Max masking: 50%"
echo "  - Warmup: 10k steps at 10% masking"
echo "=========================================="
echo ""

python -u train.py \
    -n "${experiment_name}" \
    -seed 1 \
    -config_path "config_files/STORM_AdaMAE_MaskNetTransformer_4x4.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    --use_adamae

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ckpt/${experiment_name}/"
echo "TensorBoard logs: runs/${experiment_name}/"
echo "=========================================="

