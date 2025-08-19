#!/usr/bin/env python3
"""
Comprehensive evaluation script for diversified and specialized heads experiments.

This script evaluates the performance of both diversified and specialized attention head 
masking strategies across multiple seeds and environments.

Diversified heads: [0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]
Specialized heads: [0.0, 0.08, 0.15, 0.08, 0.25, 0.15, 0.35, 0.25]

Usage:
    python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --num_seeds 5
    python eval_heads_experiments.py --env_name ALE/Pong-v5 --num_seeds 3 --episodes_per_seed 10
"""

import subprocess
import numpy as np
import pandas as pd
import json
import re
import os
import argparse
from datetime import datetime
import sys
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time


class HeadsExperimentEvaluator:
    """Evaluator for diversified and specialized heads experiments."""
    
    def __init__(self, env_name: str, trajectory_path: str, num_episodes: int = 20, timeout: int = 600):
        """
        Initialize the evaluator.
        
        Args:
            env_name: Environment name (e.g., ALE/MsPacman-v5)
            trajectory_path: Path to trajectory file (e.g., D_TRAJ/MsPacman.pkl)
            num_episodes: Number of episodes per evaluation
            timeout: Timeout in seconds for each evaluation
        """
        self.env_name = env_name
        self.trajectory_path = trajectory_path
        self.num_episodes = num_episodes
        self.timeout = timeout
        
        # Experiment configurations
        self.experiments = {
            'diversified': {
                'name': 'Diversified Heads',
                'config': 'config_files/STORM_diversified_heads.yaml',
                'masking': [0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95],
                'description': 'Evenly distributed masking across attention heads'
            },
            'specialized': {
                'name': 'Specialized Heads', 
                'config': 'config_files/STORM_specialized_heads.yaml',
                'masking': [0.0, 0.08, 0.15, 0.08, 0.25, 0.15, 0.35, 0.25],
                'description': 'Functional specialization with grouped masking percentages'
            }
        }
        
        # Extract environment short name for run naming
        self.env_short = self._extract_env_short_name(env_name)
        
    def _extract_env_short_name(self, env_name: str) -> str:
        """Extract short environment name from full environment name."""
        if 'MsPacman' in env_name:
            return 'mspacman'
        elif 'Pong' in env_name:
            return 'pong'
        elif 'Breakout' in env_name:
            return 'breakout'
        elif 'Freeway' in env_name:
            return 'freeway'
        else:
            # Fallback: extract from ALE/GameName-v5 format
            return env_name.split('/')[-1].split('-')[0].lower()
    
    def _get_run_name(self, experiment_type: str, seed: int) -> str:
        """Generate run name for experiment type and seed."""
        if experiment_type == 'diversified':
            return f"{self.env_short}-diversified-sequential-50k-seed{seed}"
        elif experiment_type == 'specialized':
            return f"{self.env_short}-specialized-sequential-50k-seed{seed}"
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def _check_checkpoint_exists(self, run_name: str) -> bool:
        """Check if checkpoint directory exists for a run."""
        checkpoint_dir = f"ckpt/{run_name}"
        return os.path.exists(checkpoint_dir)
    
    def _run_single_evaluation(self, experiment_type: str, seed: int, eval_seed: int) -> Tuple[float, str]:
        """
        Run a single evaluation and return the mean reward and any error message.
        
        Args:
            experiment_type: 'diversified' or 'specialized'
            seed: Training seed used for the model
            eval_seed: Seed for evaluation episodes
            
        Returns:
            Tuple of (mean_reward, error_message). If successful, error_message is empty.
        """
        run_name = self._get_run_name(experiment_type, seed)
        config_path = self.experiments[experiment_type]['config']
        
        # Check if checkpoint exists
        if not self._check_checkpoint_exists(run_name):
            return np.nan, f"Checkpoint directory not found: ckpt/{run_name}"
        
        cmd = [
            "python", "-u", "eval.py",
            "-env_name", self.env_name,
            "-run_name", run_name,
            "-config_path", config_path,
            "-eval_seed", str(eval_seed)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            
            # Extract mean reward from output
            for line in result.stdout.split('\n'):
                if 'Mean reward:' in line:
                    # Strip ANSI color codes and extract reward value
                    reward_text = line.split('Mean reward:')[1].strip()
                    # Remove ANSI escape sequences (colorama codes)
                    reward_text = re.sub(r'\x1b\[[0-9;]*m', '', reward_text)
                    try:
                        reward = float(reward_text)
                        return reward, ""
                    except ValueError:
                        return np.nan, f"Could not parse reward value: {reward_text}"
            
            # If no mean reward found in output
            error_msg = "No mean reward found in output"
            if result.stderr:
                error_msg += f" (stderr: {result.stderr[-200:]})"
            return np.nan, error_msg
            
        except subprocess.TimeoutExpired:
            return np.nan, f"Evaluation timed out after {self.timeout}s"
        except Exception as e:
            return np.nan, f"Evaluation failed: {str(e)}"
    
    def evaluate_experiment(self, experiment_type: str, seeds: List[int], 
                          eval_seeds: Optional[List[int]] = None) -> Dict:
        """
        Evaluate a single experiment type across multiple seeds.
        
        Args:
            experiment_type: 'diversified' or 'specialized'
            seeds: List of training seeds to evaluate
            eval_seeds: List of evaluation seeds (if None, uses seeds + 2000)
            
        Returns:
            Dictionary with evaluation results
        """
        if eval_seeds is None:
            eval_seeds = [seed + 2000 for seed in seeds]
        
        if len(eval_seeds) != len(seeds):
            raise ValueError("Number of eval_seeds must match number of seeds")
        
        experiment_info = self.experiments[experiment_type]
        print(f"\n{'='*60}")
        print(f"Evaluating {experiment_info['name']}")
        print(f"{'='*60}")
        print(f"Masking pattern: {experiment_info['masking']}")
        print(f"Description: {experiment_info['description']}")
        print(f"Seeds: {seeds}")
        print(f"Eval seeds: {eval_seeds}")
        print(f"Episodes per evaluation: {self.num_episodes}")
        print("-" * 60)
        
        results = {
            'experiment_type': experiment_type,
            'experiment_name': experiment_info['name'],
            'masking_pattern': experiment_info['masking'],
            'description': experiment_info['description'],
            'env_name': self.env_name,
            'seeds': seeds,
            'eval_seeds': eval_seeds,
            'individual_results': {},
            'rewards': [],
            'failed_evaluations': []
        }
        
        # Evaluate each seed
        for i, (seed, eval_seed) in enumerate(zip(seeds, eval_seeds), 1):
            print(f"[{i}/{len(seeds)}] Evaluating seed {seed} (eval_seed={eval_seed})...", end=" ", flush=True)
            
            reward, error = self._run_single_evaluation(experiment_type, seed, eval_seed)
            
            results['individual_results'][seed] = {
                'eval_seed': eval_seed,
                'reward': reward,
                'error': error
            }
            
            if not np.isnan(reward):
                results['rewards'].append(reward)
                print(f"✅ Reward: {reward:.2f}")
            else:
                results['failed_evaluations'].append({
                    'seed': seed,
                    'eval_seed': eval_seed,
                    'error': error
                })
                print(f"❌ Failed: {error}")
        
        # Calculate statistics
        if len(results['rewards']) > 0:
            rewards = np.array(results['rewards'])
            results['statistics'] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
                'median': float(np.median(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'count': len(rewards),
                'success_rate': len(results['rewards']) / len(seeds)
            }
            
            # Calculate 95% confidence interval
            if len(rewards) > 1:
                from scipy import stats
                confidence = 0.95
                t_score = stats.t.ppf((1 + confidence) / 2, len(rewards) - 1)
                stderr = results['statistics']['std'] / np.sqrt(len(rewards))
                margin_error = t_score * stderr
                results['statistics']['ci_lower'] = results['statistics']['mean'] - margin_error
                results['statistics']['ci_upper'] = results['statistics']['mean'] + margin_error
            else:
                results['statistics']['ci_lower'] = results['statistics']['mean']
                results['statistics']['ci_upper'] = results['statistics']['mean']
                
            print(f"\n📊 Results Summary:")
            print(f"   Mean ± Std: {results['statistics']['mean']:.2f} ± {results['statistics']['std']:.2f}")
            print(f"   95% CI: [{results['statistics']['ci_lower']:.2f}, {results['statistics']['ci_upper']:.2f}]")
            print(f"   Range: [{results['statistics']['min']:.2f}, {results['statistics']['max']:.2f}]")
            print(f"   Success rate: {results['statistics']['success_rate']:.1%} ({results['statistics']['count']}/{len(seeds)})")
            
        else:
            results['statistics'] = None
            print(f"\n❌ No successful evaluations for {experiment_info['name']}")
        
        if results['failed_evaluations']:
            print(f"\n⚠️ Failed evaluations ({len(results['failed_evaluations'])}):")
            for failure in results['failed_evaluations']:
                print(f"   Seed {failure['seed']}: {failure['error']}")
        
        return results
    
    def evaluate_all_experiments(self, seeds: List[int], 
                               eval_seeds: Optional[List[int]] = None) -> Dict:
        """
        Evaluate both diversified and specialized experiments.
        
        Args:
            seeds: List of training seeds to evaluate
            eval_seeds: List of evaluation seeds (if None, uses seeds + 2000)
            
        Returns:
            Dictionary with all evaluation results
        """
        print(f"🚀 Starting comprehensive heads experiment evaluation")
        print(f"Environment: {self.env_name}")
        print(f"Trajectory: {self.trajectory_path}")
        print(f"Seeds: {seeds}")
        print(f"Episodes per evaluation: {self.num_episodes}")
        
        all_results = {
            'evaluation_info': {
                'env_name': self.env_name,
                'trajectory_path': self.trajectory_path,
                'num_episodes': self.num_episodes,
                'timeout': self.timeout,
                'seeds': seeds,
                'eval_seeds': eval_seeds if eval_seeds else [s + 2000 for s in seeds],
                'timestamp': datetime.now().isoformat()
            },
            'experiments': {}
        }
        
        # Evaluate diversified heads
        try:
            all_results['experiments']['diversified'] = self.evaluate_experiment(
                'diversified', seeds, eval_seeds
            )
        except Exception as e:
            print(f"❌ Error evaluating diversified heads: {str(e)}")
            all_results['experiments']['diversified'] = {'error': str(e)}
        
        # Evaluate specialized heads  
        try:
            all_results['experiments']['specialized'] = self.evaluate_experiment(
                'specialized', seeds, eval_seeds
            )
        except Exception as e:
            print(f"❌ Error evaluating specialized heads: {str(e)}")
            all_results['experiments']['specialized'] = {'error': str(e)}
        
        return all_results
    
    def save_results(self, results: Dict, output_dir: str = "eval_result") -> Tuple[str, str]:
        """
        Save results to JSON and CSV files.
        
        Args:
            results: Results dictionary from evaluate_all_experiments
            output_dir: Directory to save results
            
        Returns:
            Tuple of (json_file_path, csv_file_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name_clean = self.env_short
        
        # Save detailed JSON results
        json_file = f"{output_dir}/heads_experiments_{env_name_clean}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary CSV
        csv_data = []
        for exp_type, exp_results in results['experiments'].items():
            if 'error' in exp_results:
                csv_data.append({
                    'Experiment': exp_type,
                    'Name': exp_type.capitalize(),
                    'Mean': np.nan,
                    'Std': np.nan,
                    'CI_Lower': np.nan,
                    'CI_Upper': np.nan,
                    'Count': 0,
                    'Success_Rate': 0.0,
                    'Error': exp_results['error']
                })
            elif exp_results.get('statistics'):
                stats = exp_results['statistics']
                csv_data.append({
                    'Experiment': exp_type,
                    'Name': exp_results['experiment_name'],
                    'Mean': stats['mean'],
                    'Std': stats['std'],
                    'CI_Lower': stats['ci_lower'],
                    'CI_Upper': stats['ci_upper'],
                    'Count': stats['count'],
                    'Success_Rate': stats['success_rate'],
                    'Rewards': ';'.join([f"{r:.2f}" for r in exp_results['rewards']]),
                    'Masking_Pattern': str(exp_results['masking_pattern'])
                })
            else:
                csv_data.append({
                    'Experiment': exp_type,
                    'Name': exp_results.get('experiment_name', exp_type.capitalize()),
                    'Mean': np.nan,
                    'Std': np.nan,
                    'CI_Lower': np.nan,
                    'CI_Upper': np.nan,
                    'Count': 0,
                    'Success_Rate': 0.0,
                    'Error': 'No successful evaluations'
                })
        
        csv_file = f"{output_dir}/heads_experiments_summary_{env_name_clean}_{timestamp}.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        
        return json_file, csv_file
    
    def print_comparison(self, results: Dict):
        """Print a formatted comparison of experiment results."""
        print(f"\n{'='*80}")
        print("🏆 HEADS EXPERIMENTS COMPARISON")
        print(f"{'='*80}")
        print(f"Environment: {results['evaluation_info']['env_name']}")
        print(f"Evaluation time: {results['evaluation_info']['timestamp']}")
        print("-" * 80)
        
        # Collect valid experiments
        valid_experiments = []
        for exp_type, exp_results in results['experiments'].items():
            if 'error' not in exp_results and exp_results.get('statistics'):
                valid_experiments.append((exp_type, exp_results))
        
        if not valid_experiments:
            print("❌ No experiments completed successfully!")
            return
        
        # Sort by mean performance
        valid_experiments.sort(key=lambda x: x[1]['statistics']['mean'], reverse=True)
        
        # Print header
        print(f"{'Rank':<5} {'Experiment':<15} {'Mean±Std':<15} {'95% CI':<25} {'Success':<8} {'Masking Pattern'}")
        print("-" * 80)
        
        # Print results
        for rank, (exp_type, exp_results) in enumerate(valid_experiments, 1):
            stats = exp_results['statistics']
            name = exp_results['experiment_name'][:14]
            mean_std = f"{stats['mean']:.1f}±{stats['std']:.1f}"
            ci_range = f"[{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]"
            success_rate = f"{stats['success_rate']:.0%}"
            masking = str(exp_results['masking_pattern'])[:30] + "..." if len(str(exp_results['masking_pattern'])) > 30 else str(exp_results['masking_pattern'])
            
            print(f"{rank:<5} {name:<15} {mean_std:<15} {ci_range:<25} {success_rate:<8} {masking}")
        
        print()
        
        # Statistical comparison
        if len(valid_experiments) >= 2:
            print("📈 STATISTICAL ANALYSIS")
            print("-" * 40)
            
            best_exp_type, best_exp = valid_experiments[0]
            best_stats = best_exp['statistics']
            print(f"🥇 Best: {best_exp['experiment_name']} ({best_stats['mean']:.1f} ± {best_stats['std']:.1f})")
            
            # Compare with others
            best_ci = (best_stats['ci_lower'], best_stats['ci_upper'])
            
            for exp_type, exp_results in valid_experiments[1:]:
                stats = exp_results['statistics']
                other_ci = (stats['ci_lower'], stats['ci_upper'])
                
                # Check if confidence intervals overlap
                overlap = not (best_ci[1] < other_ci[0] or other_ci[1] < best_ci[0])
                
                if not overlap:
                    diff = best_stats['mean'] - stats['mean']
                    print(f"📊 Significantly outperforms {exp_results['experiment_name']} by {diff:.1f} points")
                else:
                    print(f"📊 No significant difference vs {exp_results['experiment_name']} (overlapping CIs)")
        
        print()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate diversified and specialized heads experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate MsPacman with 5 seeds
  python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --num_seeds 5
  
  # Evaluate Pong with custom seeds
  python eval_heads_experiments.py --env_name ALE/Pong-v5 --seeds 42 43 44
  
  # Quick test with fewer episodes
  python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --num_seeds 3 --episodes 10
        """
    )
    
    parser.add_argument("--env_name", type=str, required=True,
                       help="Environment name (e.g., ALE/MsPacman-v5, ALE/Pong-v5)")
    
    parser.add_argument("--num_seeds", type=int, default=5,
                       help="Number of seeds to evaluate (default: 5)")
    
    parser.add_argument("--seeds", type=int, nargs='+',
                       help="Specific seeds to evaluate (overrides --num_seeds)")
    
    parser.add_argument("--eval_seeds", type=int, nargs='+',
                       help="Evaluation seeds (default: training_seed + 2000)")
    
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of episodes per evaluation (default: 20)")
    
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds per evaluation (default: 600)")
    
    parser.add_argument("--output_dir", type=str, default="eval_result",
                       help="Output directory for results (default: eval_result)")
    
    args = parser.parse_args()
    
    # Determine trajectory path from environment name
    trajectory_mapping = {
        'MsPacman': 'D_TRAJ/MsPacman.pkl',
        'Pong': 'D_TRAJ/Pong.pkl',
        'Freeway': 'D_TRAJ/Freeway.pkl',
        'Breakout': 'D_TRAJ/Breakout.pkl'  # Assuming it exists
    }
    
    trajectory_path = None
    for game, path in trajectory_mapping.items():
        if game in args.env_name:
            trajectory_path = path
            break
    
    if trajectory_path is None:
        print(f"❌ Error: Could not determine trajectory path for environment {args.env_name}")
        print(f"Available environments: {list(trajectory_mapping.keys())}")
        sys.exit(1)
    
    if not os.path.exists(trajectory_path):
        print(f"❌ Error: Trajectory file not found: {trajectory_path}")
        sys.exit(1)
    
    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    else:
        # Use default seeds starting from 42
        seeds = [42 + i for i in range(args.num_seeds)]
    
    # Validation
    if len(args.eval_seeds or []) > 0 and len(args.eval_seeds) != len(seeds):
        print(f"❌ Error: Number of eval_seeds ({len(args.eval_seeds)}) must match number of seeds ({len(seeds)})")
        sys.exit(1)
    
    print(f"🎯 Heads Experiments Evaluation")
    print(f"Environment: {args.env_name}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Seeds: {seeds}")
    print(f"Eval seeds: {args.eval_seeds if args.eval_seeds else [s + 2000 for s in seeds]}")
    print(f"Episodes per evaluation: {args.episodes}")
    
    try:
        # Initialize evaluator
        evaluator = HeadsExperimentEvaluator(
            env_name=args.env_name,
            trajectory_path=trajectory_path,
            num_episodes=args.episodes,
            timeout=args.timeout
        )
        
        # Run evaluation
        results = evaluator.evaluate_all_experiments(seeds, args.eval_seeds)
        
        # Save results
        json_file, csv_file = evaluator.save_results(results, args.output_dir)
        
        # Print comparison
        evaluator.print_comparison(results)
        
        # Summary
        print(f"✅ Evaluation completed!")
        print(f"📄 Detailed results: {json_file}")
        print(f"📊 Summary CSV: {csv_file}")
        
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
