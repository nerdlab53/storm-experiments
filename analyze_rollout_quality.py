#!/usr/bin/env python3
"""
Analyze how head ablation affects rollout quality in the world model.

This script examines:
1. Rollout prediction accuracy over time
2. State representation stability
3. Reward prediction coherence
4. Visual reconstruction quality

Usage:
    python analyze_rollout_quality.py -config_path config_files/STORM_monotonic_heads_0_25.yaml \
                                     -env_name ALE/MsPacman-v5 \
                                     -run_name mspacman-monotonic-heads-0_25-50k-seed42 \
                                     -rollout_length 16
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import json
from datetime import datetime
import os
from einops import rearrange

from utils import seed_np_torch, load_config
from eval import build_single_env, build_vec_env
import train
from eval_ablated_heads import set_head_ablation_in_model, clear_head_ablation_in_model


def analyze_rollout_for_head(world_model, agent, env, head_idx, rollout_length=16, num_rollouts=50):
    """Analyze rollout quality for a specific head ablation."""
    
    # Set head ablation
    if head_idx is not None:
        set_head_ablation_in_model(world_model, head_idx)
        print(f"Analyzing rollouts with head {head_idx} ablated...")
    else:
        clear_head_ablation_in_model(world_model)
        print("Analyzing baseline rollouts...")
    
    world_model.eval()
    agent.eval()
    
    metrics = {
        'head_idx': head_idx,
        'rollout_rewards': [],
        'rollout_terminations': [],
        'state_consistency': [],
        'prediction_confidence': [],
        'visual_reconstruction_error': []
    }
    
    for rollout_idx in range(num_rollouts):
        # Get initial context from environment
        obs, _ = env.reset()
        context_obs = []
        context_actions = []
        
        # Collect some real context (8 steps)
        for step in range(8):
            obs_tensor = rearrange(torch.Tensor(obs).cuda(), "H W C -> 1 1 C H W") / 255
            context_obs.append(obs_tensor)
            
            if len(context_actions) == 0:
                action = env.action_space.sample()
            else:
                # Use world model for action selection
                context_latent = world_model.encode_obs(torch.cat(context_obs, dim=1))
                model_context_action = torch.Tensor(np.array(context_actions)).cuda().unsqueeze(0)
                prior_sample, last_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                state_input = torch.cat([prior_sample, last_feat], dim=-1)
                action = agent.sample_as_env_action(state_input, greedy=True)[0]
            
            context_actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                obs, _ = env.reset()
                break
        
        # Now run imagination rollout
        with torch.no_grad():
            sample_obs = torch.cat(context_obs, dim=1)
            sample_action = torch.Tensor(np.array(context_actions)).cuda().unsqueeze(0)
            
            # Get rollout from world model
            world_model.init_imagine_buffer(1, rollout_length, dtype=torch.float32)
            world_model.storm_transformer.reset_kv_cache_list(1, dtype=torch.float32)
            
            # Process context
            context_latent = world_model.encode_obs(sample_obs)
            for i in range(sample_obs.shape[1]):
                last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = world_model.predict_next(
                    context_latent[:, i:i+1], sample_action[:, i:i+1], log_video=False
                )
            
            world_model.latent_buffer[:, 0:1] = last_latent
            world_model.hidden_buffer[:, 0:1] = last_dist_feat
            
            # Generate rollout
            rollout_rewards = []
            rollout_terminations = []
            state_norms = []
            
            for step in range(rollout_length):
                current_state = torch.cat([
                    world_model.latent_buffer[:, step:step+1], 
                    world_model.hidden_buffer[:, step:step+1]
                ], dim=-1)
                
                # Record state consistency (norm of hidden state)
                state_norms.append(current_state.norm(dim=-1).item())
                
                # Agent samples action
                action = agent.sample(current_state)
                
                # World model predicts next
                last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = world_model.predict_next(
                    world_model.latent_buffer[:, step:step+1], 
                    action, 
                    log_video=False
                )
                
                # Store predictions
                rollout_rewards.append(last_reward_hat.item())
                rollout_terminations.append(torch.sigmoid(last_termination_hat).item())
                
                # Update buffers
                if step < rollout_length - 1:
                    world_model.latent_buffer[:, step+1:step+2] = last_latent
                    world_model.hidden_buffer[:, step+1:step+2] = last_dist_feat
            
            # Store metrics for this rollout
            metrics['rollout_rewards'].append(rollout_rewards)
            metrics['rollout_terminations'].append(rollout_terminations)
            metrics['state_consistency'].append(state_norms)
            
            # Calculate prediction confidence (variance of rewards)
            reward_variance = np.var(rollout_rewards)
            metrics['prediction_confidence'].append(reward_variance)
    
    # Calculate summary statistics
    metrics['avg_reward_per_step'] = np.mean([np.mean(rewards) for rewards in metrics['rollout_rewards']])
    metrics['reward_stability'] = np.mean([np.std(rewards) for rewards in metrics['rollout_rewards']])
    metrics['avg_termination_prob'] = np.mean([np.mean(terms) for terms in metrics['rollout_terminations']])
    metrics['state_stability'] = np.mean([np.std(norms) for norms in metrics['state_consistency']])
    
    return metrics


def plot_rollout_analysis(all_metrics, output_dir):
    """Create visualizations of rollout quality across heads."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    head_indices = [m['head_idx'] for m in all_metrics]
    head_labels = ['Baseline' if h is None else f'Head {h}' for h in head_indices]
    
    avg_rewards = [m['avg_reward_per_step'] for m in all_metrics]
    reward_stabilities = [m['reward_stability'] for m in all_metrics]
    termination_probs = [m['avg_termination_prob'] for m in all_metrics]
    state_stabilities = [m['state_stability'] for m in all_metrics]
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rollout Quality Analysis Across Head Ablations', fontsize=16)
    
    # Plot 1: Average rollout rewards
    axes[0, 0].bar(head_labels, avg_rewards, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Average Imagined Reward Per Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Reward prediction stability
    axes[0, 1].bar(head_labels, reward_stabilities, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Reward Prediction Stability (Lower = More Stable)')
    axes[0, 1].set_ylabel('Std Dev of Rewards')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Termination probability
    axes[1, 0].bar(head_labels, termination_probs, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Average Predicted Termination Probability')
    axes[1, 0].set_ylabel('Termination Prob')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: State representation stability
    axes[1, 1].bar(head_labels, state_stabilities, color='gold', alpha=0.7)
    axes[1, 1].set_title('State Representation Stability (Lower = More Stable)')
    axes[1, 1].set_ylabel('Std Dev of State Norms')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rollout_quality_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed rollout trajectories plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rollout Trajectories Over Time', fontsize=16)
    
    for i, metrics in enumerate(all_metrics):
        if i >= 4:  # Only show first 4 for clarity
            break
        
        head_label = 'Baseline' if metrics['head_idx'] is None else f'Head {metrics["head_idx"]}'
        
        # Plot reward trajectories
        reward_trajectories = np.array(metrics['rollout_rewards'][:10])  # First 10 rollouts
        time_steps = range(len(reward_trajectories[0]))
        
        axes[0, 0].plot(time_steps, reward_trajectories.mean(axis=0), label=head_label, alpha=0.8)
        axes[0, 0].fill_between(time_steps, 
                               reward_trajectories.mean(axis=0) - reward_trajectories.std(axis=0),
                               reward_trajectories.mean(axis=0) + reward_trajectories.std(axis=0), 
                               alpha=0.2)
    
    axes[0, 0].set_title('Reward Predictions Over Rollout Steps')
    axes[0, 0].set_xlabel('Rollout Step')
    axes[0, 0].set_ylabel('Predicted Reward')
    axes[0, 0].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rollout_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout quality under head ablation")
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    parser.add_argument("-rollout_length", type=int, default=16)
    parser.add_argument("-num_rollouts", type=int, default=50)
    parser.add_argument("-output_dir", type=str, default="rollout_analysis")
    
    args = parser.parse_args()
    
    # Load configuration and model
    conf = load_config(args.config_path)
    seed_np_torch(seed=conf.BasicSettings.Seed)
    
    # Build environment and models
    env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
    else:
        action_dim = np.prod(env.action_space.nvec)
    
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    
    # Load checkpoint
    root_path = f"ckpt/{args.run_name}"
    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    if not pathes:
        print(f"ERROR: No checkpoint files found in {root_path}/")
        exit(1)
    
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    latest_step = max(steps)
    
    world_model.load_state_dict(torch.load(f"{root_path}/world_model_{latest_step}.pth", map_location="cuda"))
    agent.load_state_dict(torch.load(f"{root_path}/agent_{latest_step}.pth", map_location="cuda"))
    
    print(f"Loaded checkpoint from step: {latest_step}")
    
    # Analyze rollouts for baseline and each head
    all_metrics = []
    
    # Baseline
    baseline_metrics = analyze_rollout_for_head(
        world_model, agent, env, None, args.rollout_length, args.num_rollouts
    )
    all_metrics.append(baseline_metrics)
    
    # Each head
    for head_idx in range(8):
        head_metrics = analyze_rollout_for_head(
            world_model, agent, env, head_idx, args.rollout_length, args.num_rollouts
        )
        all_metrics.append(head_metrics)
    
    # Create visualizations
    plot_rollout_analysis(all_metrics, args.output_dir)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{args.output_dir}/rollout_analysis_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    for metrics in all_metrics:
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                metrics[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
    
    with open(results_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✅ Rollout analysis complete!")
    print(f"📊 Visualizations saved to: {args.output_dir}/")
    print(f"📄 Detailed results saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("ROLLOUT QUALITY SUMMARY")
    print(f"{'='*60}")
    
    for metrics in all_metrics:
        head_label = 'Baseline' if metrics['head_idx'] is None else f'Head {metrics["head_idx"]}'
        print(f"{head_label:>10}: Avg Reward: {metrics['avg_reward_per_step']:6.3f}, "
              f"Stability: {metrics['reward_stability']:6.3f}, "
              f"Term Prob: {metrics['avg_termination_prob']:6.3f}")
    
    env.close()


if __name__ == "__main__":
    main()
