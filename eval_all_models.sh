#!/bin/bash

# Evaluate all masking experiments
echo "Starting evaluation of all masking models..."

echo "=== Evaluating 0% masking model ==="
python -u eval.py \
    -env_name "ALE/Breakout-v5" \
    -run_name "breakout-mask-0p-20k-seed42" \
    -config_path "config_files/STORM_0.yaml"

echo "=== Evaluating 5% masking model ==="
python -u eval.py \
    -env_name "ALE/Breakout-v5" \
    -run_name "breakout-mask-5p-20k-seed42" \
    -config_path "config_files/STORM_5.yaml"

echo "=== Evaluating 10% masking model ==="
python -u eval.py \
    -env_name "ALE/Breakout-v5" \
    -run_name "breakout-mask-10p-20k-seed42" \
    -config_path "config_files/STORM_10.yaml"

echo "=== Evaluating 15% masking model ==="10±3%
python -u eval.py \
    -env_name "ALE/Breakout-v5" \
    -run_name "breakout-mask-15p-20k-seed42" \
    -config_path "config_files/STORM_15.yaml"

echo "=== Evaluating 25% masking model ==="
python -u eval.py \
    -env_name "ALE/Breakout-v5" \
    -run_name "breakout-mask-25p-20k-seed42" \
    -config_path "config_files/STORM_25.yaml"

echo ""
echo "All evaluations completed!"
echo "Results available in eval_result/ directory:"
echo "- breakout-mask-0p-20k-seed42.csv"
echo "- breakout-mask-5p-20k-seed42.csv" 
echo "- breakout-mask-15p-20k-seed42.csv"
echo "- breakout-mask-25p-20k-seed42.csv" 