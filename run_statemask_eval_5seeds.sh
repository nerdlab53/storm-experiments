#!/bin/bash

# StateMask Multi-Seed Evaluation Script
# 
# This script runs StateMask evaluation for the specified run across 5 seeds
# Based on the example: runs/mspacman-conservative-sequential-50k-seed42/

set -e  # Exit on any error

# Configuration
BASE_RUN_NAME="mspacman-conservative-sequential-50k-seed42"
NUM_SEEDS=5
BASE_SEED=42
ENV_NAME="ALE/MsPacman-v5"
CONFIG_PATH="config_files/STORM_masked_conservative.yaml"
OUTPUT_DIR="statemask_eval_results"
TARGET_SPARSITY=0.3

echo "🎯 StateMask Multi-Seed Evaluation"
echo "=================================="
echo "Base run: ${BASE_RUN_NAME}"
echo "Seeds: ${NUM_SEEDS} (starting from ${BASE_SEED})"
echo "Environment: ${ENV_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Target sparsity: ${TARGET_SPARSITY}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=================================="

# Check if the base run exists
RUN_DIR="runs/${BASE_RUN_NAME}"
if [ ! -d "$RUN_DIR" ]; then
    echo "❌ Error: Run directory not found: $RUN_DIR"
    echo "Please ensure the base run exists or update BASE_RUN_NAME"
    exit 1
fi

# Check if the config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "✅ Prerequisites checked"
echo ""

# Run the evaluation
echo "🚀 Starting evaluation..."
python eval_statemask_multiseed.py \
    --base_run_name "$BASE_RUN_NAME" \
    --seeds "$NUM_SEEDS" \
    --base_seed "$BASE_SEED" \
    --env_name "$ENV_NAME" \
    --config_path "$CONFIG_PATH" \
    --use_statemask \
    --target_sparsity "$TARGET_SPARSITY" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ Evaluation completed!"
echo "📁 Check results in: $OUTPUT_DIR"

# Also run without StateMask for comparison
echo ""
echo "🔄 Running baseline (no StateMask) for comparison..."
python eval_statemask_multiseed.py \
    --base_run_name "$BASE_RUN_NAME" \
    --seeds "$NUM_SEEDS" \
    --base_seed "$BASE_SEED" \
    --env_name "$ENV_NAME" \
    --config_path "$CONFIG_PATH" \
    --no_statemask \
    --output_dir "${OUTPUT_DIR}/baseline"

echo ""
echo "🎉 All evaluations completed!"
echo "📊 Results:"
echo "   StateMask results: $OUTPUT_DIR"
echo "   Baseline results: ${OUTPUT_DIR}/baseline"
