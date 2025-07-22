import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os
import scipy.stats

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_vec_env(env_name, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes_single_run(num_episode, env_name, max_steps, num_envs, image_size,
                           world_model: WorldModel, agent: agents.ActorCriticAgent, eval_seed=42, use_greedy=True):
    """Single evaluation run with specified seed"""
    world_model.eval()
    agent.eval()
    
    # Set seeds for reproducibility
    seed_np_torch(seed=eval_seed)
    
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs)
    
    sum_reward = np.zeros(num_envs)
    seeds = [eval_seed + i for i in range(num_envs)]
    current_obs, current_info = vec_env.reset(seed=seeds)
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    final_rewards = []
    
    while True:
        with torch.no_grad():
            if len(context_action) == 0:
                np.random.seed(eval_seed + len(final_rewards))
                action = vec_env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=use_greedy
                )

        context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    if len(final_rewards) == num_episode:
                        vec_env.close()
                        return final_rewards

        sum_reward += reward
        current_obs = obs
        current_info = info


def statistical_evaluation(run_name, env_name, config_path, num_runs=5, episodes_per_run=20, use_greedy=True):
    """Run multiple evaluations and compute statistics"""
    
    print(f"\n" + "="*80)
    print(f"STATISTICAL EVALUATION: {run_name}")
    print(f"Runs: {num_runs}, Episodes per run: {episodes_per_run}")
    print(f"Sampling: {'Greedy' if use_greedy else 'Stochastic'}")
    print("="*80)
    
    # Load model once
    conf = load_config(config_path)
    import train
    dummy_env = build_single_env(env_name, conf.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt/{run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[-1:]  # Use latest checkpoint
    
    for step in steps:
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        
        # Run multiple evaluations
        all_run_rewards = []
        run_means = []
        
        for run_idx in range(num_runs):
            print(f"\nRun {run_idx + 1}/{num_runs}...")
            eval_seed = 1000 + run_idx * 100  # Different seed for each run
            
            episode_rewards = eval_episodes_single_run(
                num_episode=episodes_per_run,
                env_name=env_name,
                num_envs=5,
                max_steps=conf.JointTrainAgent.SampleMaxSteps,
                image_size=conf.BasicSettings.ImageSize,
                world_model=world_model,
                agent=agent,
                eval_seed=eval_seed,
                use_greedy=use_greedy
            )
            
            run_mean = np.mean(episode_rewards)
            run_means.append(run_mean)
            all_run_rewards.extend(episode_rewards)
            
            print(f"  Run {run_idx + 1} mean: {run_mean:.3f} (episodes: {episode_rewards})")
        
        # Compute statistics
        overall_mean = np.mean(all_run_rewards)
        overall_std = np.std(all_run_rewards)
        run_means_array = np.array(run_means)
        run_mean_std = np.std(run_means_array)
        
        # 95% confidence interval for the mean
        confidence_level = 0.95
        degrees_freedom = len(run_means) - 1
        t_score = scipy.stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_error = t_score * (run_mean_std / np.sqrt(len(run_means)))
        ci_lower = np.mean(run_means) - margin_error
        ci_upper = np.mean(run_means) + margin_error
        
        print(f"\n" + "-"*60)
        print(f"RESULTS for {run_name}:")
        print(f"  Overall Mean Reward: {overall_mean:.3f} ± {overall_std:.3f}")
        print(f"  Mean across runs: {np.mean(run_means):.3f} ± {run_mean_std:.3f}")
        print(f"  95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  Individual run means: {[f'{x:.3f}' for x in run_means]}")
        print("-"*60)
        
        return {
            'run_name': run_name,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'run_means': run_means,
            'run_mean_std': run_mean_std,
            'confidence_interval': (ci_lower, ci_upper),
            'all_rewards': all_run_rewards
        }


def compare_models(models_info, env_name, num_runs=5, episodes_per_run=20):
    """Compare multiple models statistically"""
    
    results = {}
    
    for model_name, (run_name, config_path) in models_info.items():
        print(f"\nEvaluating {model_name}...")
        results[model_name] = statistical_evaluation(
            run_name=run_name,
            env_name=env_name,
            config_path=config_path,
            num_runs=num_runs,
            episodes_per_run=episodes_per_run,
            use_greedy=True
        )
    
    # Statistical comparison
    print(f"\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for model_name, result in results.items():
        ci = result['confidence_interval']
        print(f"{model_name:20s}: {result['overall_mean']:6.3f} ± {result['overall_std']:5.3f} "
              f"[{ci[0]:6.3f}, {ci[1]:6.3f}]")
    
    # Pairwise t-tests
    model_names = list(results.keys())
    if len(model_names) > 1:
        print(f"\nPairwise t-test p-values:")
        print("(H0: means are equal, H1: means are different)")
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                means1 = results[name1]['run_means']
                means2 = results[name2]['run_means']
                t_stat, p_value = scipy.stats.ttest_ind(means1, means2)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {name1} vs {name2}: p={p_value:.4f} {significance}")
    
    return results


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Define your models to compare
    models_to_compare = {
        "0% Masking": ("breakout-mask-0p-20k-seed42", "config_files/STORM_0.yaml"),
        "5% Masking": ("breakout-mask-5p-20k-seed42", "config_files/STORM_5.yaml"),
        "10% Masking": ("breakout-mask-10p-20k-seed42", "config_files/STORM_10.yaml"),
        "15% Masking": ("breakout-mask-15p-20k-seed42", "config_files/STORM_15.yaml"),
        "25% Masking" : ("breakout-mask-25p-20k-seed42", "config_files/STORM_25.yaml"),
    }
    
    # Run comparison
    results = compare_models(
        models_info=models_to_compare,
        env_name="ALE/Breakout-v5",
        num_runs=5,  # 5 independent evaluation runs
        episodes_per_run=20  # 20 episodes per run
    )
    
    # Save results
    os.makedirs('eval_result', exist_ok=True)
    with open('eval_result/statistical_comparison.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'run_name': result['run_name'],
                'overall_mean': float(result['overall_mean']),
                'overall_std': float(result['overall_std']),
                'run_means': [float(x) for x in result['run_means']],
                'run_mean_std': float(result['run_mean_std']),
                'confidence_interval': [float(result['confidence_interval'][0]), float(result['confidence_interval'][1])],
                'all_rewards': [float(x) for x in result['all_rewards']]
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to eval_result/statistical_comparison.json") 