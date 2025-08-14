#!/usr/bin/env python3

import subprocess
import numpy as np
import pandas as pd
import json
import re
import os
from datetime import datetime
import sys
from typing import Dict, List, Tuple

def run_single_evaluation(model_key: str, seed: int) -> float:
    """Run a single evaluation and return the mean reward."""
    run_name = f"mspacman-mask-{model_key}-50k-seed42"
    
    # Map model keys to config file numbers
    config_mapping = {
        "0p": "0",
        "5p": "5", 
        "10p": "10",
        "15p": "15",
        "25p": "25"
    }
    config_path = f"config_files/STORM_{config_mapping[model_key]}.yaml"
    
    cmd = [
        "python", "-u", "eval.py",
        "-env_name", "ALE/MsPacman-v5", 
        "-run_name", run_name,
        "-config_path", config_path,
        "-eval_seed", str(seed)
    ]
    
    try:
        print(f"    Running evaluation (seed={seed})...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Extract mean reward from output
        for line in result.stdout.split('\n'):
            if 'Mean reward:' in line:
                # Strip ANSI color codes and extract reward value
                import re
                reward_text = line.split('Mean reward:')[1].strip()
                # Remove ANSI escape sequences (colorama codes)
                reward_text = re.sub(r'\x1b\[[0-9;]*m', '', reward_text)
                reward = float(reward_text)
                print(f"Mean reward: {reward:.1f}")
                return reward
        
        print("ERROR: No mean reward found in output")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        print("STDERR:", result.stderr[-500:])  # Last 500 chars
        return np.nan
        
    except subprocess.TimeoutExpired:
        print("ERROR: Evaluation timed out")
        return np.nan
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return np.nan

def evaluate_all_models():
    """Evaluate all models with multiple seeds."""
    
    models = {
        "0p" : "0% Masking",
        "5p" : "5% Masking",
        "10p" : "10% Masking",
        "15p" : "15% Masking",
        "25p": "25% Masking"
    }
    
    seeds = [1001, 1002, 1003, 1004, 1005]
    num_runs = len(seeds)
    
    print("="*60)
    print("STORM Models Multi-Seed Evaluation")
    print("="*60)
    print(f"Models: {len(models)} masking variants")
    print(f"Seeds per model: {num_runs}")
    print(f"Total evaluations: {len(models) * num_runs}")
    print("="*60)
    print()
    
    # Store all results
    results = {}
    detailed_results = {}
    
    for i, (model_key, model_name) in enumerate(models.items(), 1):
        print(f"[{i}/{len(models)}] Evaluating {model_name}")
        print("-" * 50)
        
        # Check if model exists
        checkpoint_dir = f"ckpt/mspacman-mask-{model_key}-50k-seed42"
        if not os.path.exists(checkpoint_dir):
            print(f"❌ ERROR: Checkpoint directory not found: {checkpoint_dir}")
            print("   Skipping this model...")
            print()
            continue
        
        model_rewards = []
        
        for j, seed in enumerate(seeds, 1):
            print(f"  Run {j}/{num_runs} ", end="")
            reward = run_single_evaluation(model_key, seed)
            
            if not np.isnan(reward):
                model_rewards.append(reward)
            else:
                print(f"    ⚠️ Failed to get reward for seed {seed}")
        
        if len(model_rewards) > 0:
            # Calculate statistics
            mean_reward = np.mean(model_rewards)
            std_reward = np.std(model_rewards, ddof=1) if len(model_rewards) > 1 else 0.0
            
            # Calculate 95% confidence interval 
            if len(model_rewards) > 1:
                from scipy import stats
                confidence = 0.95
                t_score = stats.t.ppf((1 + confidence) / 2, len(model_rewards) - 1)
                stderr = std_reward / np.sqrt(len(model_rewards))
                margin_error = t_score * stderr
                ci_lower = mean_reward - margin_error
                ci_upper = mean_reward + margin_error
            else:
                ci_lower = ci_upper = mean_reward
            
            results[model_key] = {
                'name': model_name,
                'mean': mean_reward,
                'std': std_reward,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'count': len(model_rewards),
                'rewards': model_rewards
            }
            
            print(f"  ✅ Results: {mean_reward:.1f} ± {std_reward:.1f}")
            print(f"     95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")
            print(f"     Individual: {model_rewards}")
        else:
            print(f"  ❌ No successful evaluations for {model_name}")
            results[model_key] = {
                'name': model_name,
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'count': 0,
                'rewards': []
            }
        
        print()
    
    return results

def save_and_display_results(results: Dict):
    """Save results to file and display comparison table."""
    
    # Create results directory
    os.makedirs('eval_result', exist_ok=True)
    
    # Save detailed results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f'eval_result/multi_seed_results_{timestamp}.json'
    
    # Prepare data for JSON (convert numpy types)
    json_data = {}
    for key, data in results.items():
        json_data[key] = {
            'name': data['name'],
            'mean': float(data['mean']) if not np.isnan(data['mean']) else None,
            'std': float(data['std']) if not np.isnan(data['std']) else None,
            'ci_lower': float(data['ci_lower']) if not np.isnan(data['ci_lower']) else None,
            'ci_upper': float(data['ci_upper']) if not np.isnan(data['ci_upper']) else None,
            'count': int(data['count']),
            'rewards': [float(r) for r in data['rewards']]
        }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save summary to CSV
    csv_file = f'eval_result/multi_seed_summary_{timestamp}.csv'
    csv_data = []
    for key, data in results.items():
        csv_data.append({
            'Model': data['name'],
            'Masking': key,
            'Mean': data['mean'],
            'Std': data['std'],
            'CI_Lower': data['ci_lower'],
            'CI_Upper': data['ci_upper'],
            'Count': data['count'],
            'Rewards': ';'.join(map(str, data['rewards']))
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"📊 Detailed results saved to: {json_file}")
    print(f"📈 Summary CSV saved to: {csv_file}")
    print()
    
    # Display comparison table
    print("🏆 MODEL COMPARISON (Ranked by Mean Performance)")
    print("="*80)
    
    # Filter out failed models and sort by mean performance
    valid_results = [(key, data) for key, data in results.items() 
                    if not np.isnan(data['mean']) and data['count'] > 0]
    valid_results.sort(key=lambda x: x[1]['mean'], reverse=True)
    
    if not valid_results:
        print("❌ No models completed successfully!")
        return
    
    # Print header
    print(f"{'Rank':<4} {'Model':<15} {'Mean±Std':<12} {'95% CI':<20} {'Runs':<5} {'Individual Results'}")
    print("-" * 80)
    
    # Print results
    for rank, (key, data) in enumerate(valid_results, 1):
        mean_std = f"{data['mean']:.1f}±{data['std']:.1f}"
        ci_range = f"[{data['ci_lower']:.1f}, {data['ci_upper']:.1f}]"
        individual = ', '.join([f"{r:.1f}" for r in data['rewards']])
        
        print(f"{rank:<4} {data['name']:<15} {mean_std:<12} {ci_range:<20} {data['count']:<5} {individual}")
    
    print()
    
    # Statistical analysis
    if len(valid_results) >= 2:
        print("📈 STATISTICAL ANALYSIS")
        print("-" * 40)
        best_model = valid_results[0][1]
        print(f"🥇 Best performing: {best_model['name']} ({best_model['mean']:.1f} ± {best_model['std']:.1f})")
        
        # Check for overlapping confidence intervals
        best_ci = (best_model['ci_lower'], best_model['ci_upper'])
        
        significantly_different = []
        for key, data in valid_results[1:]:
            other_ci = (data['ci_lower'], data['ci_upper'])
            # Check if confidence intervals overlap
            overlap = not (best_ci[1] < other_ci[0] or other_ci[1] < best_ci[0])
            if not overlap:
                significantly_different.append(data['name'])
        
        if significantly_different:
            print(f"📊 Significantly outperforms: {', '.join(significantly_different)}")
        else:
            print("📊 No statistically significant differences detected (overlapping 95% CIs)")

def main():
    """Main function."""
    print("Starting comprehensive model evaluation...")
    print()
    
    try:
        results = evaluate_all_models()
        save_and_display_results(results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 