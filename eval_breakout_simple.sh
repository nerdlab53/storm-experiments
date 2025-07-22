#!/bin/bash

# Simple evaluation of all 5 masking models on Breakout
echo "🎮 Breakout Model Comparison (Simple Version)"
echo "=============================================="
echo ""

# Define models
declare -A models=(
    ["0p"]="STORM_0.yaml"
    ["5p"]="STORM_5.yaml" 
    ["10p"]="STORM_10.yaml"
    ["15p"]="STORM_15.yaml"
    ["25p"]="STORM_25.yaml"
)

declare -A model_names=(
    ["0p"]="0% Masking"
    ["5p"]="5% Masking"
    ["10p"]="10% Masking"
    ["15p"]="15% Masking" 
    ["25p"]="25% Masking"
)

# Array to store results
declare -a results=()

# Evaluate each model with 5 different seeds
for model_key in "0p" "5p" "10p" "15p" "25p"; do
    config_file="${models[$model_key]}"
    model_name="${model_names[$model_key]}"
    run_name="breakout-mask-${model_key}-20k-seed42"
    
    echo "🔍 Evaluating: $model_name"
    
    # Check if model exists
    if [ ! -d "ckpt/$run_name" ]; then
        echo "❌ Model not found: ckpt/$run_name"
        continue
    fi
    
    # Run 5 evaluations with different seeds
    scores=()
    for seed in 3001 3002 3003 3004 3005; do
        echo "  Seed $seed... " 
        
        output=$(python eval.py \
            -env_name "ALE/Breakout-v5" \
            -run_name "$run_name" \
            -config_path "config_files/$config_file" \
            -eval_seed $seed 2>&1)
        
        # Extract mean reward
        score=$(echo "$output" | grep "Mean reward:" | tail -1 | awk '{print $4}')
        
        if [[ $score =~ ^[0-9.-]+$ ]]; then
            scores+=($score)
            echo "    Score: $score"
        else
            echo "    ❌ Failed"
        fi
    done
    
    # Calculate average if we have scores
    if [ ${#scores[@]} -gt 0 ]; then
        avg=$(python3 -c "
import numpy as np
scores = [${scores[@]}]
print(f'{np.mean(scores):.3f}')
")
        std=$(python3 -c "
import numpy as np  
scores = [${scores[@]}]
print(f'{np.std(scores):.3f}')
")
        
        echo "  📊 Average: $avg ± $std"
        results+=("$avg:$model_name:$std")
    else
        echo "  ❌ No valid scores"
    fi
    echo ""
done

# Display final ranking
echo "🏆 FINAL RESULTS"
echo "================"

if [ ${#results[@]} -gt 0 ]; then
    # Sort by score (descending)
    IFS=$'\n' sorted=($(sort -t: -k1 -nr <<<"${results[*]}"))
    
    echo "Rank | Model           | Score"
    echo "-----|-----------------|--------"
    
    for i in "${!sorted[@]}"; do
        IFS=':' read -r score model_name std <<< "${sorted[$i]}"
        rank=$((i + 1))
        
        if [ $rank -eq 1 ]; then
            emoji="🥇"
        elif [ $rank -eq 2 ]; then
            emoji="🥈" 
        elif [ $rank -eq 3 ]; then
            emoji="🥉"
        else
            emoji="  "
        fi
        
        printf "%s %2d | %-15s | %s ± %s\n" "$emoji" "$rank" "$model_name" "$score" "$std"
    done
    
    # Winner
    IFS=':' read -r best_score best_model best_std <<< "${sorted[0]}"
    echo ""
    echo "🎉 WINNER: $best_model ($best_score ± $best_std)"
    
else
    echo "❌ No results obtained!"
fi 