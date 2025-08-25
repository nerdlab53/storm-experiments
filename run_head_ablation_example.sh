#!/bin/bash

# Example script for running head ablation experiments
# Modify the parameters below according to your specific setup

echo "=== Head Ablation Experiment ==="
echo "Testing individual head contributions to model performance"
echo ""

# Monotonic heads experiment (0-25% masking range)
echo "Running monotonic heads ablation..."
python eval_ablated_heads.py \
    -config_path config_files/STORM_monotonic_heads_0_25.yaml \
    -env_name ALE/MsPacman-v5 \
    -run_name mspacman-monotonic-heads-0_25-50k-seed42 \
    -num_seeds 3

echo ""
echo "Monotonic heads ablation completed!"

echo ""
echo "=== Head ablation experiment finished! ==="
echo "Check the eval_result/ directory for JSON results files."
echo ""
echo "Expected output structure:"
echo "- Baseline performance for each seed"
echo "- Individual head ablation results"
echo "- Impact statistics (mean ± std) for each head"
echo ""
echo "Monotonic masking pattern: [0.0, 0.0357, 0.0714, 0.1071, 0.1429, 0.1786, 0.2143, 0.25]"
echo "This tests whether gradual masking increase produces different head importance patterns"
echo "compared to functional specialization grouping."