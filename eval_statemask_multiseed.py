#!/usr/bin/env python3
"""
StateMask Multi-Seed Evaluation Script

This script runs StateMask evaluation experiments across multiple seeds and 
collects comprehensive statistics on performance, blinding behavior, and 
fidelity metrics.

Usage:
    python eval_statemask_multiseed.py --base_run_name "mspacman-conservative-sequential-50k-seed42" --seeds 5
    python eval_statemask_multiseed.py --base_run_name "pong-statemask-experiment-seed42" --seeds 10
"""

import subprocess
import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
import yaml
from pathlib import Path

# Add project imports
sys.path.append('.')
from statemask_trainer import create_statemask_trainer
from utils import load_config
import agents
from train import build_world_model, build_agent


class StateMaskEvaluator:
    """Comprehensive StateMask evaluation across multiple seeds."""
    
    def __init__(self, base_run_name: str, config_path: str, env_name: str = "ALE/MsPacman-v5"):
        self.base_run_name = base_run_name
        self.config_path = config_path
        self.env_name = env_name
        self.config = load_config(config_path)
        
        # Results storage
        self.results = {
            'performance': [],
            'statemask_metrics': [],
            'blinding_statistics': [],
            'training_curves': []
        }
        
        # Initialize components for StateMask evaluation
        self.feat_dim = 32*32 + self.config['Models']['WorldModel']['TransformerHiddenDim']
        
        print(f"🎯 StateMask Evaluator Initialized")
        print(f"   Base run: {base_run_name}")
        print(f"   Environment: {env_name}")
        print(f"   Feature dim: {self.feat_dim}")
        print(f"   Config: {config_path}")
    
    def load_trained_model(self, run_path: str) -> Tuple[agents.ActorCriticAgent, torch.nn.Module]:
        """Load trained world model and agent from run directory."""
        
        # Load model checkpoint
        checkpoint_path = os.path.join(run_path, "model.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Build models
        world_model = build_world_model(self.config)
        agent = build_agent(self.config, action_dim=6)  # Assuming Atari games
        
        # Load state dicts
        if 'world_model_state_dict' in checkpoint:
            world_model.load_state_dict(checkpoint['world_model_state_dict'])
        if 'agent_state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['agent_state_dict'])
        
        # Move to device
        world_model = world_model.cuda()
        agent = agent.cuda()
        
        return agent, world_model
    
    def run_single_evaluation(self, seed: int, use_statemask: bool = True, 
                            statemask_config: Optional[Dict] = None) -> Dict:
        """Run evaluation for a single seed."""
        
        print(f"  🔸 Evaluating seed {seed} (StateMask: {'ON' if use_statemask else 'OFF'})")
        
        # Determine run path
        run_path = f"runs/{self.base_run_name}"
        if not os.path.exists(run_path):
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        
        try:
            # Load trained models
            agent, world_model = self.load_trained_model(run_path)
            
            # Create StateMask if requested
            statemask = None
            statemask_trainer = None
            
            if use_statemask:
                if statemask_config is None:
                    statemask_config = {
                        'hidden_dim': 128,
                        'lr': 1e-4,
                        'fidelity_weight': 1.0,
                        'sparsity_weight': 0.1,
                        'target_sparsity': 0.3,
                        'train_frequency': 100,
                        'batch_size': 256
                    }
                
                statemask, statemask_trainer = create_statemask_trainer(
                    feat_dim=self.feat_dim,
                    config=statemask_config
                )
                statemask = statemask.cuda()
            
            # Run performance evaluation using eval.py
            perf_result = self._run_performance_eval(seed, use_statemask)
            
            # Run StateMask-specific evaluations
            statemask_metrics = {}
            blinding_stats = {}
            
            if use_statemask and statemask is not None:
                statemask_metrics, blinding_stats = self._evaluate_statemask_behavior(
                    agent, world_model, statemask, statemask_trainer, seed
                )
            
            return {
                'seed': seed,
                'use_statemask': use_statemask,
                'performance': perf_result,
                'statemask_metrics': statemask_metrics,
                'blinding_stats': blinding_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    ❌ Error evaluating seed {seed}: {e}")
            return {
                'seed': seed,
                'use_statemask': use_statemask,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _run_performance_eval(self, seed: int, use_statemask: bool) -> Dict:
        """Run standard performance evaluation."""
        
        env_short = "mspacman" if "MsPacman" in self.env_name else "pong" if "Pong" in self.env_name else "breakout"
        
        cmd = [
            "python", "-u", "eval.py",
            "-env_name", self.env_name,
            "-run_name", self.base_run_name,
            "-config_path", self.config_path,
            "-eval_seed", str(seed)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse results from output
            mean_reward = None
            episode_count = None
            
            for line in result.stdout.split('\n'):
                if 'Mean reward:' in line:
                    # Extract reward value
                    import re
                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    try:
                        mean_reward = float(clean_line.split('Mean reward:')[1].strip())
                    except (IndexError, ValueError):
                        pass
                
                if 'Episodes:' in line or 'Total episodes:' in line:
                    try:
                        episode_count = int(line.split(':')[1].strip())
                    except (IndexError, ValueError):
                        pass
            
            return {
                'mean_reward': mean_reward,
                'episode_count': episode_count,
                'success': mean_reward is not None,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-500:] if result.stderr else ""
            }
            
        except subprocess.TimeoutExpired:
            return {'error': 'Evaluation timeout', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def _evaluate_statemask_behavior(self, agent, world_model, statemask, statemask_trainer, seed: int) -> Tuple[Dict, Dict]:
        """Evaluate StateMask-specific behavior and metrics."""
        
        # Generate synthetic evaluation data
        batch_size = 64
        seq_len = 16
        
        # Create random observation batch (normally would use real evaluation data)
        obs_batch = torch.randn(batch_size, seq_len, 3, 64, 64).cuda()
        action_batch = torch.randint(0, 6, (batch_size, seq_len)).cuda()
        
        with torch.no_grad():
            # Encode observations
            encoded_obs = world_model.encode_obs(obs_batch)
            
            # Get transformer hidden states
            world_model.storm_transformer.reset_kv_cache_list(batch_size, dtype=torch.float32)
            
            hidden_states = []
            for i in range(seq_len):
                _, _, _, _, hidden = world_model.predict_next(
                    encoded_obs[:, i:i+1],
                    action_batch[:, i:i+1] if i < seq_len-1 else torch.zeros_like(action_batch[:, 0:1]),
                    log_video=False
                )
                hidden_states.append(hidden)
            
            # Combine features
            final_hidden = hidden_states[-1]
            final_latent = encoded_obs[:, -1:]
            combined_features = torch.cat([final_latent, final_hidden], dim=-1).squeeze(1)
            
            # Get policy logits
            policy_logits = agent.policy(combined_features)
            
            # Evaluate StateMask
            statemask_metrics = statemask_trainer.evaluate_masking_quality(
                combined_features, policy_logits
            )
            
            # Get blinding statistics
            blinding_stats = statemask_trainer.get_blinding_statistics(combined_features)
            
            # Additional StateMask analysis
            gate_probs = statemask(combined_features)
            
            # Action distribution analysis
            original_dist = torch.softmax(policy_logits, dim=-1)
            
            # Simulate masked actions
            binary_masks = torch.bernoulli(gate_probs)
            num_actions = policy_logits.shape[-1]
            uniform_dist = torch.full_like(original_dist, 1.0 / num_actions)
            
            # Mixed distribution (what agent would see with masking)
            mixed_dist = binary_masks * original_dist + (1 - binary_masks) * uniform_dist
            
            # Calculate distribution divergences
            kl_divergence = torch.nn.functional.kl_div(
                mixed_dist.log(), original_dist, reduction='batchmean'
            )
            
            additional_metrics = {
                'gate_prob_histogram': gate_probs.cpu().numpy().flatten().tolist(),
                'action_entropy_original': -(original_dist * original_dist.log()).sum(dim=-1).mean().item(),
                'action_entropy_mixed': -(mixed_dist * mixed_dist.log()).sum(dim=-1).mean().item(),
                'kl_divergence': kl_divergence.item(),
                'effective_masking_rate': (1 - binary_masks.mean()).item()
            }
            
            # Combine metrics
            statemask_metrics.update(additional_metrics)
        
        return statemask_metrics, blinding_stats
    
    def run_multiseed_evaluation(self, num_seeds: int = 5, base_seed: int = 42, 
                                use_statemask: bool = True, statemask_config: Optional[Dict] = None) -> Dict:
        """Run evaluation across multiple seeds."""
        
        print(f"🚀 Starting Multi-Seed StateMask Evaluation")
        print(f"   Seeds: {num_seeds} (starting from {base_seed})")
        print(f"   StateMask: {'Enabled' if use_statemask else 'Disabled'}")
        print(f"   Base run: {self.base_run_name}")
        print("="*60)
        
        all_results = []
        
        for i in range(num_seeds):
            seed = base_seed + i
            print(f"🔸 Evaluating seed {seed} ({i+1}/{num_seeds})")
            
            result = self.run_single_evaluation(seed, use_statemask, statemask_config)
            all_results.append(result)
            
            # Print progress
            if 'performance' in result and result['performance'].get('success', False):
                reward = result['performance'].get('mean_reward', 'N/A')
                print(f"    ✅ Reward: {reward}")
                
                if use_statemask and 'statemask_metrics' in result:
                    blinding_rate = result['blinding_stats'].get('blinding_rate', 'N/A')
                    fidelity = result['statemask_metrics'].get('eval_statemask/fidelity_loss', 'N/A')
                    print(f"    📊 Blinding: {blinding_rate:.3f}, Fidelity Loss: {fidelity:.4f}")
            else:
                print(f"    ❌ Evaluation failed")
            print()
        
        # Aggregate results
        summary = self._aggregate_results(all_results, use_statemask)
        
        return {
            'summary': summary,
            'individual_results': all_results,
            'config': {
                'num_seeds': num_seeds,
                'base_seed': base_seed,
                'use_statemask': use_statemask,
                'statemask_config': statemask_config,
                'base_run_name': self.base_run_name,
                'env_name': self.env_name
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_results(self, results: List[Dict], use_statemask: bool) -> Dict:
        """Aggregate results across seeds."""
        
        # Extract successful results
        successful_results = [r for r in results if r.get('performance', {}).get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful evaluations'}
        
        # Performance metrics
        rewards = [r['performance']['mean_reward'] for r in successful_results]
        episodes = [r['performance']['episode_count'] for r in successful_results if r['performance']['episode_count'] is not None]
        
        summary = {
            'performance': {
                'mean_reward_mean': np.mean(rewards),
                'mean_reward_std': np.std(rewards),
                'mean_reward_min': np.min(rewards),
                'mean_reward_max': np.max(rewards),
                'successful_seeds': len(successful_results),
                'total_seeds': len(results)
            }
        }
        
        if episodes:
            summary['performance'].update({
                'episodes_mean': np.mean(episodes),
                'episodes_std': np.std(episodes)
            })
        
        # StateMask metrics (if applicable)
        if use_statemask:
            statemask_results = [r for r in successful_results if 'statemask_metrics' in r and r['statemask_metrics']]
            
            if statemask_results:
                # Aggregate StateMask metrics
                metrics_keys = statemask_results[0]['statemask_metrics'].keys()
                statemask_summary = {}
                
                for key in metrics_keys:
                    values = [r['statemask_metrics'][key] for r in statemask_results if key in r['statemask_metrics']]
                    if values and all(isinstance(v, (int, float)) for v in values):
                        statemask_summary[f'{key}_mean'] = np.mean(values)
                        statemask_summary[f'{key}_std'] = np.std(values)
                
                summary['statemask'] = statemask_summary
                
                # Aggregate blinding statistics
                blinding_results = [r for r in successful_results if 'blinding_stats' in r and r['blinding_stats']]
                if blinding_results:
                    blinding_keys = blinding_results[0]['blinding_stats'].keys()
                    blinding_summary = {}
                    
                    for key in blinding_keys:
                        values = [r['blinding_stats'][key] for r in blinding_results if key in r['blinding_stats']]
                        if values and all(isinstance(v, (int, float)) for v in values):
                            blinding_summary[f'{key}_mean'] = np.mean(values)
                            blinding_summary[f'{key}_std'] = np.std(values)
                    
                    summary['blinding'] = blinding_summary
        
        return summary
    
    def save_results(self, results: Dict, output_dir: str = "statemask_eval_results"):
        """Save evaluation results to files."""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"statemask_eval_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary CSV
        if 'summary' in results and 'performance' in results['summary']:
            csv_path = os.path.join(output_dir, f"statemask_summary_{timestamp}.csv")
            summary_df = pd.DataFrame([results['summary']['performance']])
            summary_df.to_csv(csv_path, index=False)
        
        # Save detailed results CSV
        if 'individual_results' in results:
            detail_path = os.path.join(output_dir, f"statemask_detailed_{timestamp}.csv")
            detailed_data = []
            
            for result in results['individual_results']:
                row = {
                    'seed': result.get('seed'),
                    'use_statemask': result.get('use_statemask'),
                    'mean_reward': result.get('performance', {}).get('mean_reward'),
                    'episode_count': result.get('performance', {}).get('episode_count'),
                    'success': result.get('performance', {}).get('success', False)
                }
                
                # Add StateMask metrics if available
                if 'statemask_metrics' in result and result['statemask_metrics']:
                    for key, value in result['statemask_metrics'].items():
                        if isinstance(value, (int, float)):
                            row[f'statemask_{key}'] = value
                
                if 'blinding_stats' in result and result['blinding_stats']:
                    for key, value in result['blinding_stats'].items():
                        if isinstance(value, (int, float)):
                            row[f'blinding_{key}'] = value
                
                detailed_data.append(row)
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(detail_path, index=False)
        
        print(f"📁 Results saved to:")
        print(f"   📄 {json_path}")
        print(f"   📊 {csv_path if 'summary' in results else 'No summary CSV'}")
        print(f"   📈 {detail_path if 'individual_results' in results else 'No detailed CSV'}")
        
        return json_path


def main():
    parser = argparse.ArgumentParser(description="StateMask Multi-Seed Evaluation")
    parser.add_argument("--base_run_name", type=str, required=True,
                       help="Base run name (e.g., 'mspacman-conservative-sequential-50k-seed42')")
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of evaluation seeds to run")
    parser.add_argument("--base_seed", type=int, default=42,
                       help="Starting seed for evaluation")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5",
                       help="Environment name")
    parser.add_argument("--config_path", type=str, default="config_files/STORM_masked_conservative.yaml",
                       help="Path to config file")
    parser.add_argument("--use_statemask", action="store_true", default=True,
                       help="Enable StateMask evaluation")
    parser.add_argument("--no_statemask", action="store_true",
                       help="Disable StateMask evaluation")
    parser.add_argument("--target_sparsity", type=float, default=0.3,
                       help="Target sparsity for StateMask")
    parser.add_argument("--output_dir", type=str, default="statemask_eval_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Handle statemask flags
    use_statemask = args.use_statemask and not args.no_statemask
    
    # StateMask configuration
    statemask_config = {
        'hidden_dim': 128,
        'lr': 1e-4,
        'fidelity_weight': 1.0,
        'sparsity_weight': 0.1,
        'target_sparsity': args.target_sparsity,
        'train_frequency': 100,
        'batch_size': 256
    }
    
    print("🎯 StateMask Multi-Seed Evaluation")
    print("="*50)
    print(f"Base run: {args.base_run_name}")
    print(f"Seeds: {args.seeds} (starting from {args.base_seed})")
    print(f"Environment: {args.env_name}")
    print(f"Config: {args.config_path}")
    print(f"StateMask: {'Enabled' if use_statemask else 'Disabled'}")
    if use_statemask:
        print(f"Target sparsity: {args.target_sparsity}")
    print("="*50)
    
    # Create evaluator
    evaluator = StateMaskEvaluator(
        base_run_name=args.base_run_name,
        config_path=args.config_path,
        env_name=args.env_name
    )
    
    # Run evaluation
    results = evaluator.run_multiseed_evaluation(
        num_seeds=args.seeds,
        base_seed=args.base_seed,
        use_statemask=use_statemask,
        statemask_config=statemask_config if use_statemask else None
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    if 'summary' in results and 'performance' in results['summary']:
        perf = results['summary']['performance']
        print(f"🎯 Performance:")
        print(f"   Mean Reward: {perf['mean_reward_mean']:.2f} ± {perf['mean_reward_std']:.2f}")
        print(f"   Range: [{perf['mean_reward_min']:.2f}, {perf['mean_reward_max']:.2f}]")
        print(f"   Successful: {perf['successful_seeds']}/{perf['total_seeds']} seeds")
        
        if use_statemask and 'statemask' in results['summary']:
            print(f"\n🎭 StateMask Behavior:")
            sm = results['summary']['statemask']
            for key, value in sm.items():
                if 'mean' in key:
                    base_key = key.replace('_mean', '')
                    std_key = key.replace('_mean', '_std')
                    std_val = sm.get(std_key, 0)
                    print(f"   {base_key}: {value:.4f} ± {std_val:.4f}")
    
    # Save results
    output_path = evaluator.save_results(results, args.output_dir)
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
