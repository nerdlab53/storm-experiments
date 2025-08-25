#!/bin/bash

# Enhanced head ablation experiment with rollout efficiency metrics
# This version includes additional metrics for better understanding

echo "=== Enhanced Head Ablation Experiment ==="
echo "Testing individual head contributions with efficiency metrics"
echo ""

# Monotonic heads experiment (0-25% masking range)
echo "Running enhanced monotonic heads ablation..."
python eval_ablated_heads.py \
    -config_path config_files/STORM_monotonic_heads_0_25.yaml \
    -env_name ALE/MsPacman-v5 \
    -run_name mspacman-monotonic-heads-0_25-50k-seed42 \
    -num_seeds 3

echo ""
echo "Enhanced head ablation completed!"

echo ""
echo "=== Enhanced experiment finished! ==="
echo "Check the eval_result/ directory for detailed JSON results files."
echo ""
echo "New metrics included:"
echo "- Episode length and efficiency (reward per step)"
echo "- Q-value range (decision confidence)"
echo "- Action entropy (policy uncertainty)"
echo "- Action diversity (exploration behavior)"
echo "- Detailed impact assessment for each head"
echo ""
echo "Monotonic masking pattern: [0.0, 0.0357, 0.0714, 0.1071, 0.1429, 0.1786, 0.2143, 0.25]"
