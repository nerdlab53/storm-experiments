#!/bin/bash

# Evaluation script for Base STORM and AdaMAE across 5 seeds
# Usage: bash eval_both_models.sh

env_name="Breakout"
config_base="config_files/STORM.yaml"
config_adamae="config_files/STORM_AdaMAE.yaml"

seeds=(1 2 3 4 5)

echo "=========================================="
echo "Evaluating Base STORM and AdaMAE models"
echo "Environment: ${env_name}"
echo "Seeds: ${seeds[@]}"
echo "=========================================="

# Evaluate Base STORM
echo ""
echo "Evaluating Base STORM..."
for seed in "${seeds[@]}"; do
    run_name="${env_name}-life_done-base-75k-seed${seed}"
    echo "  Seed ${seed}: ${run_name}"
    python -u eval.py \
        -env_name "ALE/${env_name}-v5" \
        -run_name "${run_name}" \
        -config_path "${config_base}" \
        -eval_seed 42
    
    if [ $? -ne 0 ]; then
        echo "  WARNING: Evaluation failed for ${run_name}"
    fi
done

# Evaluate AdaMAE
echo ""
echo "Evaluating AdaMAE..."
for seed in "${seeds[@]}"; do
    run_name="${env_name}-life_done-AdaMAE-75k-seed${seed}"
    echo "  Seed ${seed}: ${run_name}"
    python -u eval.py \
        -env_name "ALE/${env_name}-v5" \
        -run_name "${run_name}" \
        -config_path "${config_adamae}" \
        -eval_seed 42 \
        --use_adamae
    
    if [ $? -ne 0 ]; then
        echo "  WARNING: Evaluation failed for ${run_name}"
    fi
done

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to eval_result/"
echo "=========================================="
