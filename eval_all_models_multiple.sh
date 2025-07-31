#!/bin/bash

# Evaluate all masking experiments with multiple seeds
echo "Starting multi-seed evaluation of all masking models..."
echo "Each model will be evaluated 5 times with different seeds"
echo ""

# Define models and their display names
declare -A models=(
    ["0p"]="0% Masking"
    ["5p"]="5% Masking" 
    ["10p"]="10% Masking"
    ["15p"]="15% Masking"
    ["25p"]="25% Masking"
)

# Create results directory
mkdir -p eval_result

# Results summary file
summary_file="eval_result/multi_seed_summary.txt"
echo "Multi-Seed Evaluation Results" > $summary_file
echo "=============================" >> $summary_file
echo "" >> $summary_file

for model_key in "0p" "5p" "10p" "15p" "25p"; do
    model_name="${models[$model_key]}"
    echo "=== Evaluating $model_name ==="
    echo "=== Evaluating $model_name ===" >> $summary_file
    
    # Array to collect results
    results=()
    
    for run in {1..5}; do
        seed=$((1000 + run))
        echo "  Run $run/5 (seed=$seed)..."
        
        # Run evaluation
        output=$(python -u eval.py \
            -env_name "ALE/MsPacman-v5" \
            -run_name "mspacman-mask-${model_key}-20k-seed42" \
            -config_path "config_files/STORM_${model_key}.yaml" \
            -eval_seed $seed 2>&1)
        
        # Extract mean reward
        mean_reward=$(echo "$output" | grep "Mean reward:" | tail -1 | awk '{print $4}')
        results+=($mean_reward)
        echo "    Mean reward: $mean_reward"
    done
    
    # Calculate statistics using Python
    stats=$(python3 -c "
import numpy as np
import scipy.stats as stats

results = np.array([${results[@]}])
mean = np.mean(results)
std = np.std(results, ddof=1)  # Sample std
stderr = std / np.sqrt(len(results))

# 95% confidence interval
confidence = 0.95
t_score = stats.t.ppf((1 + confidence) / 2, len(results) - 1)
margin_error = t_score * stderr
ci_lower = mean - margin_error
ci_upper = mean + margin_error

print(f'{mean:.3f},{std:.3f},{ci_lower:.3f},{ci_upper:.3f}')
print(f'Individual: {list(results)}')
")
    
    # Parse statistics
    IFS=',' read -r mean std ci_lower ci_upper <<< "$(echo "$stats" | head -1)"
    individual_line=$(echo "$stats" | tail -1)
    
    echo "  Results: ${results[@]}"
    echo "  Mean ± Std: $mean ± $std"
    echo "  95% CI: [$ci_lower, $ci_upper]"
    echo ""
    
    # Save to summary file
    echo "Model: $model_name" >> $summary_file
    echo "  Mean ± Std: $mean ± $std" >> $summary_file
    echo "  95% CI: [$ci_lower, $ci_upper]" >> $summary_file
    echo "  $individual_line" >> $summary_file
    echo "" >> $summary_file
done

echo "All evaluations completed!"
echo "Summary saved to: $summary_file"
echo ""
echo "Quick comparison:"
echo "=================="

# Display final comparison
python3 -c "
import re

# Read summary file
with open('$summary_file', 'r') as f:
    content = f.read()

# Extract results
models = []
for match in re.finditer(r'Model: (.+?)\n.*?Mean ± Std: ([0-9.]+) ± ([0-9.]+)\n.*?95% CI: \[([0-9.]+), ([0-9.]+)\]', content, re.DOTALL):
    name, mean, std, ci_low, ci_high = match.groups()
    models.append((name, float(mean), float(std), float(ci_low), float(ci_high)))

# Sort by mean performance
models.sort(key=lambda x: x[1], reverse=True)

print('Rank | Model         | Mean±Std  | 95% CI')
print('-----|---------------|-----------|------------------')
for i, (name, mean, std, ci_low, ci_high) in enumerate(models, 1):
    print(f'{i:4d} | {name:13s} | {mean:5.3f}±{std:.3f} | [{ci_low:.3f}, {ci_high:.3f}]')
" 