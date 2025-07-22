#!/bin/bash

# Evaluate all 5 masking models on Breakout
echo "🎮 BREAKOUT MODEL COMPARISON"
echo "============================"
echo "Evaluating 5 masking models with multiple seeds each..."
echo ""

# Define models and their config files
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

# Create results directory
mkdir -p eval_result

# Results file
results_file="eval_result/breakout_comparison_results.txt"
echo "Breakout Model Comparison Results" > $results_file
echo "=================================" >> $results_file
echo "Date: $(date)" >> $results_file
echo "" >> $results_file

# Array to store final results for ranking
declare -a final_results=()

# Evaluate each model
for model_key in "0p" "5p" "10p" "15p" "25p"; do
    config_file="${models[$model_key]}"
    model_name="${model_names[$model_key]}"
    run_name="breakout-mask-${model_key}-20k-seed42"
    
    echo "🔍 Evaluating: $model_name"
    echo "   Config: config_files/$config_file"
    echo "   Model: $run_name"
    echo ""
    
    # Save to results file
    echo "=== $model_name ===" >> $results_file
    echo "Config: $config_file" >> $results_file
    echo "Model: $run_name" >> $results_file
    echo "" >> $results_file
    
    # Check if model exists
    if [ ! -d "ckpt/$run_name" ]; then
        echo "❌ ERROR: Model checkpoint not found at ckpt/$run_name"
        echo "   Skipping $model_name..."
        echo "ERROR: Model not found" >> $results_file
        echo "" >> $results_file
        continue
    fi
    
    # Run evaluation with multiple seeds and episodes
    echo "   Running evaluation (3 seed runs, 30 episodes each)..."
    python eval.py \
        -env_name "ALE/Breakout-v5" \
        -run_name "$run_name" \
        -config_path "config_files/$config_file" \
        -num_episodes 30 \
        -num_seed_runs 3 \
        -eval_seed 2000
    
    # Check if evaluation was successful
    if [ $? -eq 0 ]; then
        echo "✅ $model_name evaluation completed"
        
        # Extract results from detailed JSON file
        detailed_file="eval_result/${run_name}_detailed.json"
        if [ -f "$detailed_file" ]; then
            # Extract overall mean and std using Python
            result=$(python3 -c "
import json
try:
    with open('$detailed_file', 'r') as f:
        data = json.load(f)
    step_data = list(data.values())[0]  # Get latest checkpoint
    mean = step_data['overall_mean']
    std = step_data['overall_std']
    episodes = step_data['total_episodes']
    print(f'{mean:.3f},{std:.3f},{episodes}')
except Exception as e:
    print('ERROR,ERROR,ERROR')
")
            
            IFS=',' read -r mean std episodes <<< "$result"
            
            if [ "$mean" != "ERROR" ]; then
                echo "   📊 Result: $mean ± $std (${episodes} episodes)"
                echo "Result: $mean ± $std (${episodes} episodes)" >> $results_file
                
                # Store for final ranking
                final_results+=("$mean:$model_name:$std")
            else
                echo "   ❌ Failed to extract results"
                echo "Failed to extract results" >> $results_file
            fi
        else
            echo "   ❌ Detailed results file not found"
            echo "Detailed results file not found" >> $results_file
        fi
    else
        echo "❌ $model_name evaluation failed"
        echo "Evaluation failed" >> $results_file
    fi
    
    echo "" >> $results_file
    echo ""
done

# Rank results
echo "🏆 FINAL RANKING"
echo "================"

if [ ${#final_results[@]} -gt 0 ]; then
    # Sort results by mean score (descending)
    IFS=$'\n' sorted=($(sort -t: -k1 -nr <<<"${final_results[*]}"))
    
    echo "Rank | Model           | Mean ± Std" >> $results_file
    echo "-----|-----------------|------------" >> $results_file
    
    echo "Rank | Model           | Mean ± Std"
    echo "-----|-----------------|------------"
    
    for i in "${!sorted[@]}"; do
        IFS=':' read -r mean model_name std <<< "${sorted[$i]}"
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
        
        printf "%s %2d | %-15s | %6s ± %s\n" "$emoji" "$rank" "$model_name" "$mean" "$std"
        printf "%4d | %-15s | %6s ± %s\n" "$rank" "$model_name" "$mean" "$std" >> $results_file
    done
    
    # Declare winner
    IFS=':' read -r best_score best_model best_std <<< "${sorted[0]}"
    echo ""
    echo "🎉 WINNER: $best_model"
    echo "   Score: $best_score ± $best_std"
    echo ""
    echo "" >> $results_file
    echo "WINNER: $best_model" >> $results_file
    echo "Score: $best_score ± $best_std" >> $results_file
    
else
    echo "❌ No successful evaluations completed!"
    echo "ERROR: No successful evaluations" >> $results_file
fi

echo "📁 Results saved to: $results_file"
echo ""
echo "📋 Individual model details available in:"
echo "   eval_result/breakout-mask-*_detailed.json" 