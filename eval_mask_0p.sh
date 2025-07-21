#!/bin/bash

# Evaluation script for 0% masking model
echo "Evaluating 0% masking model..."

python -u eval.py \
    -env_name "ALE/Pong-v5" \
    -run_name "pong-mask-0p-20k-seed42" \
    -config_path "config_files/STORM_25.yaml"

echo "Evaluation of 0% masking model completed!"
echo "Results saved to: eval_result/pong-mask-0p-20k-seed42.csv" 