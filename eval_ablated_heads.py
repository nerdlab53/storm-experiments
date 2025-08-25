#!/usr/bin/env python3
"""
Minimal evaluation script for head ablation experiments.

This script sequentially ablates each of the 8 attention heads and evaluates
performance over multiple seeds to understand individual head contributions.

Usage:
    python eval_ablated_heads.py -config_path config_files/STORM_specialized_heads.yaml \
                                 -env_name ALE/MsPacman-v5 \
                                 -run_name mspacman-specialized-sequential-50k-seed42 \
                                 -num_seeds 3
"""

import argparse
import numpy as np
import torch
import colorama
import os
import glob
from tqdm import tqdm
import json
from datetime import datetime
from collections import deque
from einops import rearrange
import torch.nn.functional as F

from utils import seed_np_torch, load_config
from eval import build_single_env, build_vec_env
import train


def set_head_ablation_in_model(world_model, head_idx):
    """Set head ablation for a specific head index in the world model."""
    # Access the transformer's attention blocks
    transformer = world_model.storm_transformer
    for layer in transformer.layer_stack:
        if hasattr(layer, 'mh_attn'):
            layer.mh_attn.set_head_ablation(head_idx)


def clear_head_ablation_in_model(world_model):
    """Clear head ablation from the world model."""
    transformer = world_model.storm_transformer
    for layer in transformer.layer_stack:
        if hasattr(layer, 'mh_attn'):
            layer.mh_attn.clear_head_ablation()


def evaluate_single_head_ablation(world_model, agent, args, conf, head_idx=None):
    """
    Evaluate model with a specific head ablated (or baseline if head_idx=None).
    
    Args:
        world_model: The world model to evaluate
        agent: The agent to evaluate
        args: Command line arguments
        conf: Configuration object
        head_idx: Index of head to ablate (0-7), or None for baseline
    
    Returns:
        Dictionary with performance and efficiency metrics
    """
    # Set or clear head ablation
    if head_idx is not None:
        set_head_ablation_in_model(world_model, head_idx)
        print(f"  Ablating head {head_idx}")
    else:
        clear_head_ablation_in_model(world_model)
        print("  Baseline (no ablation)")
    
    # Run enhanced evaluation
    metrics = eval_episodes_with_metrics(
        num_episode=20,  # Fixed number of episodes
        env_name=args.env_name,
        max_steps=200000,
        num_envs=1,
        image_size=conf.BasicSettings.ImageSize,
        world_model=world_model,
        agent=agent,
        eval_seed=args.eval_seed
    )
    
    return metrics


def eval_episodes_with_metrics(num_episode, env_name, max_steps, num_envs, image_size,
                              world_model, agent, eval_seed=42):
    """Enhanced evaluation with rollout efficiency metrics."""
    world_model.eval()
    agent.eval()
    
    # Set evaluation seed for reproducible episodes
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs)
    
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    
    # Properly seed the vector environment
    seeds = [eval_seed + i for i in range(num_envs)]
    current_obs, current_info = vec_env.reset(seed=seeds)
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # Tracking variables for metrics
    final_rewards = []
    episode_lengths = []
    q_value_ranges = []
    action_entropies = []
    episode_actions = []
    current_episode_length = 0
    current_episode_actions = []
    
    while True:
        current_episode_length += 1
        
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                np.random.seed(eval_seed + len(final_rewards))
                action = vec_env.action_space.sample()
                q_range = 0.0  
                # Handle both Discrete and MultiDiscrete action spaces
                if hasattr(vec_env.action_space, 'n'):
                    action_entropy = np.log(vec_env.action_space.n)  # Max entropy for uniform
                else:
                    # For MultiDiscrete, calculate entropy based on total number of actions
                    action_entropy = np.log(np.prod(vec_env.action_space.nvec))
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                
                state_input = torch.cat([prior_flattened_sample, last_dist_feat], dim=-1)
                logits = agent.policy(state_input)
                
                q_range = (logits.max(dim=-1)[0] - logits.min(dim=-1)[0]).mean().item()
                probs = F.softmax(logits, dim=-1)
                action_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                
                action = agent.sample_as_env_action(state_input, greedy=True)

        # Store metrics
        q_value_ranges.append(q_range)
        action_entropies.append(action_entropy)
        current_episode_actions.append(action[0] if isinstance(action, np.ndarray) else action)
        
        context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    episode_lengths.append(current_episode_length)
                    episode_actions.append(current_episode_actions.copy())
                    
                    # Reset for next episode
                    sum_reward[i] = 0
                    current_episode_length = 0
                    current_episode_actions = []
                    
                    if len(final_rewards) == num_episode:
                        vec_env.close()
                        
                        # Calculate final metrics
                        mean_reward = np.mean(final_rewards)
                        mean_episode_length = np.mean(episode_lengths)
                        reward_per_step = mean_reward / mean_episode_length if mean_episode_length > 0 else 0
                        mean_q_range = np.mean(q_value_ranges) if q_value_ranges else 0
                        mean_action_entropy = np.mean(action_entropies) if action_entropies else 0
                        
                        # Action consistency metrics
                        all_actions = np.concatenate(episode_actions)
                        # Handle both Discrete and MultiDiscrete action spaces
                        if hasattr(vec_env.action_space, 'n'):
                            total_actions = vec_env.action_space.n
                        else:
                            total_actions = np.prod(vec_env.action_space.nvec)
                        action_diversity = len(np.unique(all_actions)) / total_actions
                        
                        print("Mean reward: " + colorama.Fore.YELLOW + f"{mean_reward:.2f}" + colorama.Style.RESET_ALL)
                        print(f"  Episode length: {mean_episode_length:.1f}, Reward/step: {reward_per_step:.2f}")
                        print(f"  Q-range: {mean_q_range:.2f}, Action entropy: {mean_action_entropy:.2f}")
                        print(f"  Action diversity: {action_diversity:.2f}")
                        
                        return {
                            'mean_reward': mean_reward,
                            'episode_length': mean_episode_length,
                            'reward_per_step': reward_per_step,
                            'q_value_range': mean_q_range,
                            'action_entropy': mean_action_entropy,
                            'action_diversity': action_diversity,
                            'episode_rewards': final_rewards,
                            'episode_lengths': episode_lengths
                        }

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Evaluate head ablation effects on model performance"
    )
    parser.add_argument("-config_path", type=str, required=True,
                       help="Path to model configuration file")
    parser.add_argument("-env_name", type=str, required=True,
                       help="Environment name (e.g., ALE/MsPacman-v5)")
    parser.add_argument("-run_name", type=str, required=True,
                       help="Run name for checkpoint loading")
    parser.add_argument("-num_seeds", type=int, default=3,
                       help="Number of seeds to evaluate (default: 3)")
    parser.add_argument("-episodes_per_eval", type=int, default=20,
                       help="Number of episodes per evaluation (default: 20)")
    parser.add_argument("-output_dir", type=str, default="eval_result",
                       help="Output directory for results (default: eval_result)")
    
    args = parser.parse_args()
    
    # Load configuration
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set base seed
    seed_np_torch(seed=conf.BasicSettings.Seed)
    
    # Build model and agent
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
    # Handle both Discrete and MultiDiscrete action spaces
    if hasattr(dummy_env.action_space, 'n'):
        action_dim = dummy_env.action_space.n
    else:
        action_dim = np.prod(dummy_env.action_space.nvec)
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    dummy_env.close()
    
    # Find checkpoint
    root_path = f"ckpt/{args.run_name}"
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    if not pathes:
        print(f"ERROR: No checkpoint files found in {root_path}/")
        exit(1)
    
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    latest_step = steps[-1]
    print(f"Using checkpoint from step: {latest_step}")
    
    # Load checkpoint
    world_model.load_state_dict(torch.load(f"{root_path}/world_model_{latest_step}.pth", map_location="cuda"))
    agent.load_state_dict(torch.load(f"{root_path}/agent_{latest_step}.pth", map_location="cuda"))
    
    # Results storage
    results = {
        'config_path': args.config_path,
        'env_name': args.env_name,
        'run_name': args.run_name,
        'num_seeds': args.num_seeds,
        'episodes_per_eval': args.episodes_per_eval,
        'checkpoint_step': latest_step,
        'timestamp': datetime.now().isoformat(),
        'seed_results': {}
    }
    
    print(f"\n{'='*60}")
    print(f"HEAD ABLATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Environment: {args.env_name}")
    print(f"Run name: {args.run_name}")
    print(f"Config: {args.config_path}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Episodes per evaluation: {args.episodes_per_eval}")
    print("-" * 60)
    
    # Evaluate across seeds
    for seed_idx in range(args.num_seeds):
        eval_seed = 42 + seed_idx * 1000  # Different eval seeds
        args.eval_seed = eval_seed
        
        print(f"\n[SEED {seed_idx + 1}/{args.num_seeds}] Eval seed: {eval_seed}")
        print("-" * 40)
        
        seed_results = {
            'eval_seed': eval_seed,
            'baseline': None,
            'ablated_heads': {}
        }
        
        # 1. Baseline evaluation (no ablation)
        print("Evaluating baseline...")
        baseline_metrics = evaluate_single_head_ablation(
            world_model, agent, args, conf, head_idx=None
        )
        seed_results['baseline'] = baseline_metrics
        baseline_reward = baseline_metrics['mean_reward']
        print(f"  → Baseline reward: {baseline_reward:.2f}")
        
        # 2. Ablate each head sequentially
        for head_idx in range(8):  # 8 heads total
            print(f"Evaluating with head {head_idx} ablated...")
            ablated_metrics = evaluate_single_head_ablation(
                world_model, agent, args, conf, head_idx=head_idx
            )
            seed_results['ablated_heads'][head_idx] = ablated_metrics
            
            # Calculate impact
            ablated_reward = ablated_metrics['mean_reward']
            impact = baseline_reward - ablated_reward
            
            # Enhanced reporting
            print(f"  → Ablated reward: {ablated_reward:.2f} (impact: {impact:+.2f})")
            print(f"    Episode efficiency: {ablated_metrics['reward_per_step']:.3f} vs baseline {baseline_metrics['reward_per_step']:.3f}")
            print(f"    Decision confidence: Q-range {ablated_metrics['q_value_range']:.2f} vs baseline {baseline_metrics['q_value_range']:.2f}")
        
        results['seed_results'][seed_idx] = seed_results
    
    # Calculate summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    
    # Collect all baseline results
    baseline_rewards = [results['seed_results'][i]['baseline']['mean_reward'] for i in range(args.num_seeds)]
    baseline_q_ranges = [results['seed_results'][i]['baseline']['q_value_range'] for i in range(args.num_seeds)]
    baseline_efficiency = [results['seed_results'][i]['baseline']['reward_per_step'] for i in range(args.num_seeds)]
    
    summary = {
        'baseline_stats': {
            'mean_reward': float(np.mean(baseline_rewards)),
            'reward_std': float(np.std(baseline_rewards, ddof=1)) if len(baseline_rewards) > 1 else 0.0,
            'q_range_mean': float(np.mean(baseline_q_ranges)),
            'efficiency_mean': float(np.mean(baseline_efficiency)),
            'values': baseline_rewards
        },
        'head_impacts': {}
    }
    
    print(f"Baseline Performance: {summary['baseline_stats']['mean_reward']:.2f} ± {summary['baseline_stats']['reward_std']:.2f}")
    print(f"Baseline Q-Range: {summary['baseline_stats']['q_range_mean']:.2f}")
    print(f"Baseline Efficiency: {summary['baseline_stats']['efficiency_mean']:.3f}")
    print("\nHead Impact Summary:")
    print("-" * 80)
    print("Head | Reward Impact | Q-Range Impact | Efficiency Impact | Overall Assessment")
    print("-" * 80)
    
    for head_idx in range(8):
        head_rewards = [results['seed_results'][i]['ablated_heads'][head_idx]['mean_reward'] for i in range(args.num_seeds)]
        head_q_ranges = [results['seed_results'][i]['ablated_heads'][head_idx]['q_value_range'] for i in range(args.num_seeds)]
        head_efficiency = [results['seed_results'][i]['ablated_heads'][head_idx]['reward_per_step'] for i in range(args.num_seeds)]
        
        reward_impacts = [baseline_rewards[i] - head_rewards[i] for i in range(args.num_seeds)]
        q_range_impacts = [head_q_ranges[i] - baseline_q_ranges[i] for i in range(args.num_seeds)]
        efficiency_impacts = [head_efficiency[i] - baseline_efficiency[i] for i in range(args.num_seeds)]
        
        reward_impact_mean = float(np.mean(reward_impacts))
        q_range_impact_mean = float(np.mean(q_range_impacts))
        efficiency_impact_mean = float(np.mean(efficiency_impacts))
        
        # Determine overall assessment
        if reward_impact_mean > 50:
            assessment = "CRITICAL"
        elif reward_impact_mean > 10:
            assessment = "Important"
        elif reward_impact_mean < -50:
            assessment = "HARMFUL"
        elif reward_impact_mean < -10:
            assessment = "Problematic"
        else:
            assessment = "Neutral"
        
        summary['head_impacts'][head_idx] = {
            'reward_impact_mean': reward_impact_mean,
            'reward_impact_std': float(np.std(reward_impacts, ddof=1)) if len(reward_impacts) > 1 else 0.0,
            'q_range_impact_mean': q_range_impact_mean,
            'efficiency_impact_mean': efficiency_impact_mean,
            'assessment': assessment,
            'ablated_values': head_rewards
        }
        
        print(f" {head_idx:3d} | {reward_impact_mean:+8.1f}     | {q_range_impact_mean:+9.2f}     | {efficiency_impact_mean:+11.3f}     | {assessment:>10s}")
    
    results['summary'] = summary
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = args.env_name.split('/')[-1].split('-')[0].lower()
    results_file = f"{args.output_dir}/head_ablation_{env_short}_{args.run_name.split('-')[1]}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Clear any remaining ablation
    clear_head_ablation_in_model(world_model)


if __name__ == "__main__":
    # Ignore warnings and optimize CUDA
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main()
