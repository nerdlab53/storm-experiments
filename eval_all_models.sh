#!/bin/bash

# Evaluate all masking experiments
echo "Starting evaluation of all masking models..."

echo "=== Evaluating 0% masking model ==="
python -u eval.py \
    -env_name "ALE/Pong-v5" \
    -run_name "pong-mask-0p-20k-seed42" \
    -config_path "config_files/STORM_25.yaml"

echo "=== Evaluating 5% masking model ==="
python -u eval.py \
    -env_name "ALE/Pong-v5" \
    -run_name "pong-mask-5p-20k-seed42" \
    -config_path "config_files/STORM_25.yaml"

echo "=== Evaluating 15% masking model ==="
python -u eval.py \
    -env_name "ALE/Pong-v5" \
    -run_name "pong-mask-15p-20k-seed42" \
    -config_path "config_files/STORM_25.yaml"

echo "=== Evaluating 25% masking model ==="
python -u eval.py \
    -env_name "ALE/Pong-v5" \
    -run_name "pong-mask-25p-20k-seed42" \
    -config_path "config_files/STORM_25.yaml"

echo ""
echo "All evaluations completed!"
echo "Results available in eval_result/ directory:"
echo "- pong-mask-0p-20k-seed42.csv"
echo "- pong-mask-5p-20k-seed42.csv" 
echo "- pong-mask-15p-20k-seed42.csv"
echo "- pong-mask-25p-20k-seed42.csv" 