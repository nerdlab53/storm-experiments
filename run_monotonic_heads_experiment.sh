#!/bin/bash

# Experiment: Monotonic vs Specialized Head Masking
# Tests whether the specific functional grouping matters or just the masking range (0-35%)

echo "=== STORM Head Masking Range Experiment ==="
echo "Comparing monotonic (0-35%) vs specialized grouping patterns"
echo "Hypothesis: If performance is similar, then range matters more than specific pattern"
echo ""

# Create experiment directory
mkdir -p experiments/head_masking_range_test
cd experiments/head_masking_range_test

echo "Starting monotonic heads experiment..."
echo "Masking pattern: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]"
echo "Training for 50k steps..."

# Run monotonic masking experiment
python ../../main.py --config ../../config_files/STORM_monotonic_heads.yaml \
    --logdir ./monotonic_heads_logs \
    --device cuda \
    2>&1 | tee monotonic_heads_training.log

echo ""
echo "Monotonic heads experiment completed!"
echo "Results saved in: experiments/head_masking_range_test/monotonic_heads_logs"
echo ""
echo "To compare with specialized heads, run the specialized experiment:"
echo "python ../../main.py --config ../../config_files/STORM_specialized_heads.yaml --logdir ./specialized_heads_logs"
echo ""
echo "=== Experimental Analysis ==="
echo "Expected outcomes:"
echo "1. Similar performance → Range (0-35%) is key factor"
echo "2. Worse performance → Functional grouping/redundancy matters"
echo "3. Better performance → Monotonic might be superior!"
