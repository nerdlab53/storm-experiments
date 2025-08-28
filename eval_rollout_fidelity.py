#!/usr/bin/env python3
"""
Long-horizon rollout fidelity and error growth evaluation for STORM.

Measures compounding error when rolling out closed-loop inside the world model
for horizons k ∈ {1,2,4,8,16,32,64}, compares against ground-truth trajectories
generated with a frozen behavior policy π_b, and supports ablations:
 - deterministic latent sampling (mode)
 - limited attention context (truncate KV cache to last N tokens)
 - no reward head / no termination head (ignore those predictions)

Outputs JSON and CSV with metrics per horizon and ablation, including agent
rewards (ground-truth returns under π_b and predicted returns in-model).
"""

import argparse
import os
import json
import time
import glob
from collections import deque, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils import load_config, seed_np_torch
import train
from eval import build_single_env, build_vec_env


@torch.no_grad()
def collect_ground_truth_trajectories(env_name: str,
                                      image_size: int,
                                      agent: torch.nn.Module,
                                      world_model,
                                      num_episodes: int = 20,
                                      context_len: int = 16,
                                      device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> List[Dict]:
    """
    Collect trajectories using the frozen behavior policy π_b (the trained agent)
    with the same state feature construction used during training and eval
    (concat of flattened latent and last transformer hidden).

    Returns a list of episodes, each containing:
      - obs: np.ndarray [T, H, W, C]
      - actions: np.ndarray [T]
      - rewards: np.ndarray [T]
    """
    vec_env = build_vec_env(env_name, image_size, num_envs=1)
    episodes = []

    for ep in range(num_episodes):
        context_obs = deque(maxlen=context_len)
        context_action = deque(maxlen=context_len)
        obs, info = vec_env.reset()
        done = np.array([False])
        truncated = np.array([False])
        ep_obs: List[np.ndarray] = []
        ep_actions: List[int] = []
        ep_rewards: List[float] = []

        while True:
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                # Build combined state as in training
                obs_tensor = torch.tensor(np.stack(list(context_obs), axis=1), dtype=torch.float32, device=device) / 255.0
                obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3)  # B L C H W
                context_latent = world_model.encode_obs(obs_tensor)
                model_context_action = torch.tensor(np.stack(list(context_action), axis=1), dtype=torch.float32, device=device)
                prior_flattened, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                combined_state = torch.cat([prior_flattened, last_dist_feat], dim=-1)
                action = agent.sample_as_env_action(combined_state, greedy=True)

            ep_obs.append(obs[0])
            ep_actions.append(int(action[0]))
            context_obs.append(np.expand_dims(obs[0], axis=0))  # shape [1,H,W,C]
            context_action.append(action)

            obs, reward, done, trunc, info = vec_env.step(action)
            truncated = trunc
            ep_rewards.append(float(reward[0]))

            if (done | truncated)[0]:
                episodes.append({
                    'obs': np.stack(ep_obs, axis=0),
                    'actions': np.array(ep_actions, dtype=np.int64),
                    'rewards': np.array(ep_rewards, dtype=np.float32)
                })
                break

    vec_env.close()
    return episodes


def truncate_kv_cache(transformer, max_len: int):
    """Limit attention context by truncating KV cache to last max_len tokens."""
    if not hasattr(transformer, 'kv_cache_list') or transformer.kv_cache_list is None:
        return
    for i in range(len(transformer.kv_cache_list)):
        kv = transformer.kv_cache_list[i]
        if kv is not None and kv.shape[1] > max_len:
            transformer.kv_cache_list[i] = kv[:, -max_len:, :]


@torch.no_grad()
def closed_loop_rollout_from_state(world_model,
                                   agent,
                                   start_obs_hwcn: np.ndarray,
                                   horizons: List[int],
                                   ablation: Dict,
                                   device: torch.device,
                                   collect_seq: bool = False) -> Dict[int, Dict]:
    """
    Roll out inside the world model from a real observation (state) start_obs_hwcn.
    Returns a dict mapping horizon k -> predictions dict with:
      - obs_hat: decoded pixels at horizon k (or None if decoder disabled)
      - latent: flattened latent at horizon k
      - prior_logits: logits of prior at horizon k
      - reward_hat_seq: predicted rewards for steps 1..k
    """
    world_model.eval()

    # Seed sequence with start_obs latent and a dummy action (alignment with encoder interface)
    obs_t = torch.tensor(start_obs_hwcn, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)  # [1,1,H,W,C]
    obs_t = obs_t.permute(0, 1, 4, 2, 3) / 255.0  # [1,1,C,H,W]
    seq_latent = world_model.encode_obs(obs_t)  # [1,1,stoch]
    seq_action = torch.zeros((1, 1), dtype=torch.float32, device=device)  # [1,1]

    outputs: Dict[int, Dict] = {}
    reward_hat_seq: List[float] = []
    pred_seq: List[np.ndarray] = []
    k_max = max(horizons)
    deterministic = ablation.get('deterministic', False)
    limit_ctx = ablation.get('limit_ctx', 0)
    use_reward_head = ablation.get('use_reward_head', True)
    use_termination_head = ablation.get('use_termination_head', True)

    for step in range(1, k_max + 1):
        # Compute current transformer state and next prior latent using full-seq path
        if limit_ctx and limit_ctx > 0:
            seq_lat_for_attn = seq_latent[:, -limit_ctx:, :]
            seq_act_for_attn = seq_action[:, -limit_ctx:]
        else:
            seq_lat_for_attn = seq_latent
            seq_act_for_attn = seq_action
        prior_flattened, dist_feat_last = world_model.calc_last_dist_feat(seq_lat_for_attn, seq_act_for_attn)
        combined_state = torch.cat([prior_flattened, dist_feat_last], dim=-1)
        action = agent.sample(combined_state, greedy=True)  # [1,1]
        
        # Decode predictions for logging/metrics
        prior_logits = world_model.dist_head.forward_prior(dist_feat_last)
        if deterministic:
            sample = world_model.stright_throught_gradient(prior_logits, sample_mode="mode")
            next_latent = world_model.flatten_sample(sample)
        else:
            # prior_flattened already contains sampled next latent
            next_latent = prior_flattened
        obs_hat = world_model.image_decoder(next_latent)
        reward_logits = world_model.reward_decoder(dist_feat_last)
        reward_hat = world_model.symlog_twohot_loss_func.decode(reward_logits)
        termination_hat = world_model.termination_decoder(dist_feat_last)

        # Disable heads if requested
        if not use_reward_head:
            reward_hat = torch.zeros_like(reward_hat)
        if not use_termination_head:
            termination_hat = torch.zeros_like(termination_hat)

        # Accumulate rewards
        if use_reward_head:
            reward_hat_seq.append(float(reward_hat.squeeze().detach().cpu().item()))
        # Accumulate frames for video
        if collect_seq and obs_hat is not None:
            frame = obs_hat.detach().cpu().float().squeeze(0).squeeze(0).permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0.0, 1.0)
            pred_seq.append(frame)

        # Save outputs if step in horizons
        if step in horizons:
            outputs[step] = {
                'obs_hat': None if obs_hat is None else obs_hat.detach().cpu().float().squeeze(0).squeeze(0).permute(1, 2, 0).numpy(),  # HWC
                'latent': next_latent.detach().cpu(),
                'prior_logits': prior_logits.detach().cpu(),
                'reward_hat_seq': reward_hat_seq.copy()
            }

        # Advance sequence
        seq_latent = torch.cat([seq_latent, next_latent], dim=1)
        seq_action = torch.cat([seq_action, action], dim=1)

    if collect_seq:
        # Store full predicted sequence up to k_max as an entry at key 0
        outputs[0] = {'pred_seq': pred_seq}
    return outputs


def set_head_ablation_in_model(world_model, head_idx: int):
    """Enable ablation for a specific attention head across all layers."""
    transformer = world_model.storm_transformer
    for layer in transformer.layer_stack:
        if hasattr(layer, 'slf_attn') and hasattr(layer.slf_attn, 'set_head_ablation'):
            layer.slf_attn.set_head_ablation(head_idx)


def clear_head_ablation_in_model(world_model):
    transformer = world_model.storm_transformer
    for layer in transformer.layer_stack:
        if hasattr(layer, 'slf_attn') and hasattr(layer.slf_attn, 'clear_head_ablation'):
            layer.slf_attn.clear_head_ablation()


def write_side_by_side_video(pred_seq: List[np.ndarray], gt_seq: List[np.ndarray], path: str, fps: int = 10):
    try:
        import imageio.v2 as imageio
    except Exception:
        imageio = None
    # Convert to uint8 and side-by-side stack
    frames = []
    length = min(len(pred_seq), len(gt_seq))
    for i in range(length):
        pred = np.clip(pred_seq[i] * 255.0, 0, 255).astype(np.uint8)
        gt = np.clip(gt_seq[i].astype(np.float32), 0, 255).astype(np.uint8)
        # Ensure same size
        if pred.shape != gt.shape:
            H = min(pred.shape[0], gt.shape[0])
            W = min(pred.shape[1], gt.shape[1])
            pred = pred[:H, :W]
            gt = gt[:H, :W]
        canvas = np.concatenate([gt, pred], axis=1)
        frames.append(canvas)
    if imageio is not None:
        imageio.mimwrite(path, frames, fps=fps, macro_block_size=None)
    else:
        # Fallback: save as NPZ if video writer not available
        np.savez_compressed(path.replace('.mp4', '.npz'), frames=np.stack(frames, axis=0))


def compute_pixel_mse(pred: np.ndarray, target: np.ndarray) -> float:
    # pred in [0,1], target in [0,255] HWC
    pred = (np.clip(pred, 0.0, 1.0) * 255.0).astype(np.float32)
    target = target.astype(np.float32)
    return float(np.mean((pred - target) ** 2))


def latent_ce_and_kl(prior_logits: torch.Tensor, post_logits: torch.Tensor) -> Tuple[float, float]:
    prior = torch.distributions.OneHotCategorical(logits=prior_logits)
    post = torch.distributions.OneHotCategorical(logits=post_logits)
    # Cross-entropy H(post, prior) = H(post) + KL(post||prior)
    # Approximate with mean over batch/length/code dims
    probs_post = post.probs
    log_probs_prior = prior.logits.log_softmax(dim=-1)
    ce = -(probs_post * log_probs_prior).sum(dim=-1).mean().item()
    kl = torch.distributions.kl.kl_divergence(post, prior).mean().item()
    return float(ce), float(kl)


def calibration_ece(prior_logits: torch.Tensor, post_logits: torch.Tensor, num_bins: int = 15) -> float:
    probs = prior_logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    true = post_logits.softmax(dim=-1).argmax(dim=-1)
    conf = conf.flatten().cpu().numpy()
    correct = (pred == true).flatten().cpu().numpy().astype(np.float32)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.any():
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            ece += (mask.sum() / len(conf)) * abs(acc - avg_conf)
    return float(ece)


def evaluate(args):
    seed_np_torch(args.seed)
    conf = load_config(args.config_path)

    # Build models
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    # Force CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_model = world_model.to(device)
    agent = agent.to(device)
    dummy_env.close()

    # Load latest checkpoint
    root_path = f"ckpt/{args.run_name}"
    ckpts = glob.glob(os.path.join(root_path, "world_model_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {root_path}")
    steps = sorted([int(os.path.basename(p).split('_')[-1].split('.')[0]) for p in ckpts])
    latest = steps[-1]
    world_model.load_state_dict(torch.load(os.path.join(root_path, f"world_model_{latest}.pth"), map_location=device))
    agent.load_state_dict(torch.load(os.path.join(root_path, f"agent_{latest}.pth"), map_location=device))
    world_model.eval(); agent.eval()

    # Collect ground-truth episodes under frozen π_b
    episodes = collect_ground_truth_trajectories(
        env_name=args.env_name,
        image_size=conf.BasicSettings.ImageSize,
        agent=agent,
        world_model=world_model,
        num_episodes=args.gt_episodes,
        context_len=args.context_len,
        device=device
    )

    horizons = sorted(list(set(args.horizons)))
    ablation_settings = {
        'full': dict(deterministic=False, limit_ctx=0, use_reward_head=True, use_termination_head=True),
        'deterministic': dict(deterministic=True, limit_ctx=0, use_reward_head=True, use_termination_head=True),
        'limit_ctx': dict(deterministic=False, limit_ctx=args.limit_ctx, use_reward_head=True, use_termination_head=True),
        'no_reward': dict(deterministic=False, limit_ctx=0, use_reward_head=False, use_termination_head=True),
        'no_termination': dict(deterministic=False, limit_ctx=0, use_reward_head=True, use_termination_head=False),
    }

    # Metrics storage
    results: Dict = {
        'env_name': args.env_name,
        'run_name': args.run_name,
        'checkpoint_step': latest,
        'horizons': horizons,
        'ablations': list(ablation_settings.keys()),
        'seed': args.seed,
        'metrics': {abl: {k: defaultdict(list) for k in horizons} for abl in ablation_settings.keys()},
        'agent_rewards': {
            'episode_returns': [float(ep['rewards'].sum()) for ep in episodes],
            'mean_return': float(np.mean([ep['rewards'].sum() for ep in episodes]))
        }
    }

    # Iterate over episodes and timesteps
    head_indices = list(range(world_model.num_heads)) if hasattr(world_model, 'num_heads') else list(range(8))
    eval_sets = [('baseline', None)] + [(f'head_{h}', h) for h in head_indices]

    for label, head_idx in eval_sets:
        if head_idx is None:
            clear_head_ablation_in_model(world_model)
        else:
            set_head_ablation_in_model(world_model, head_idx)

        for ep in episodes:
            obs_seq = ep['obs']  # [T,H,W,C]
            rew_seq = ep['rewards']  # [T]
            T_len = len(obs_seq)
            # Sample up to max_pairs per episode
            t_indices = np.linspace(0, max(0, T_len - max(horizons) - 1), num=min(args.max_pairs_per_ep, max(1, T_len - max(horizons))), dtype=int)
            for t in t_indices:
                start_obs = obs_seq[t]
                # For each ablation, roll out in-model
                for abl_name, abl_conf in ablation_settings.items():
                    outputs = closed_loop_rollout_from_state(world_model, agent, start_obs, horizons, abl_conf, device, collect_seq=True)
                    for k in horizons:
                        gt_idx = min(t + k, T_len - 1)
                        gt_obs = obs_seq[gt_idx]
                        # Pixel MSE
                        if outputs[k]['obs_hat'] is not None:
                            mse = compute_pixel_mse(outputs[k]['obs_hat'], gt_obs)
                            results['metrics'][abl_name][k]['pixel_mse'].append(mse)
                        # Latent metrics (CE, KL)
                        # Posterior from ground-truth obs at t+k
                        obs_tk = torch.tensor(gt_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1).permute(0, 1, 4, 2, 3) / 255.0
                        post_logits = world_model.dist_head.forward_post(world_model.encoder(obs_tk))  # [1,1,K,C]
                        prior_logits = outputs[k]['prior_logits']  # [1,1,K,C] on CPU
                        ce, kl = latent_ce_and_kl(prior_logits.to(device), post_logits)
                        results['metrics'][abl_name][k]['latent_ce'].append(ce)
                        results['metrics'][abl_name][k]['latent_kl'].append(kl)
                        # Calibration (ECE)
                        ece = calibration_ece(prior_logits.to(device), post_logits)
                        results['metrics'][abl_name][k]['ece'].append(ece)
                        # Rewards: ground-truth cumulative and predicted cumulative
                        gt_return_k = float(rew_seq[t:gt_idx].sum())
                        results['metrics'][abl_name][k]['gt_return'].append(gt_return_k)
                        pred_return_k = float(np.sum(outputs[k]['reward_hat_seq'])) if len(outputs[k]['reward_hat_seq']) > 0 else 0.0
                        results['metrics'][abl_name][k]['pred_return'].append(pred_return_k)

                        # Video logging: write once per (head, ablation, t) for the max horizon
                        if args.save_videos and k == max(horizons) and 0 in outputs:
                            pred_seq = outputs[0].get('pred_seq', [])
                            # Build GT sequence: s_{t+1..t+k}
                            gt_seq = [obs_seq[min(t + i, T_len - 1)] for i in range(1, k + 1)]
                            os.makedirs(args.video_dir, exist_ok=True)
                            fname = f"{label}_{abl_name}_t{int(t)}_k{k}.mp4"
                            path = os.path.join(args.video_dir, fname)
                            write_side_by_side_video(pred_seq, gt_seq, path, fps=10)

    # Aggregate statistics
    summary = {}
    for abl_name in ablation_settings.keys():
        summary[abl_name] = {}
        for k in horizons:
            bucket = results['metrics'][abl_name][k]
            summary[abl_name][k] = {m: float(np.mean(vals)) if len(vals) > 0 else float('nan') for m, vals in bucket.items()}
            # Also store std for pixel_mse and returns
            for key in ['pixel_mse', 'latent_ce', 'latent_kl', 'ece', 'gt_return', 'pred_return']:
                if key in bucket and len(bucket[key]) > 1:
                    summary[abl_name][k][key + '_std'] = float(np.std(bucket[key], ddof=1))

    results['summary'] = summary

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = os.path.join(args.output_dir, f"rollout_fidelity_{args.run_name}_{stamp}")
    with open(base + '.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Also write a simple CSV for quick viewing
    try:
        import csv
        with open(base + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['ablation', 'horizon', 'pixel_mse', 'latent_ce', 'latent_kl', 'ece', 'gt_return', 'pred_return']
            writer.writerow(header)
            for abl_name in ablation_settings.keys():
                for k in horizons:
                    s = summary[abl_name][k]
                    writer.writerow([
                        abl_name, k,
                        s.get('pixel_mse', ''), s.get('latent_ce', ''), s.get('latent_kl', ''), s.get('ece', ''),
                        s.get('gt_return', ''), s.get('pred_return', '')
                    ])
    except Exception:
        pass

    print(f"Saved results to {base}.json and {base}.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Long-horizon rollout fidelity evaluation")
    parser.add_argument('-config_path', type=str, required=True)
    parser.add_argument('-env_name', type=str, required=True)
    parser.add_argument('-run_name', type=str, required=True)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('--gt_episodes', type=int, default=10)
    parser.add_argument('--context_len', type=int, default=16)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--limit_ctx', type=int, default=4, help='Context length for limited attention ablation')
    parser.add_argument('--max_pairs_per_ep', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='eval_result')
    parser.add_argument('--save_videos', action='store_true', help='Save side-by-side GT vs predicted videos')
    parser.add_argument('--video_dir', type=str, default='eval_result/videos', help='Directory to save videos')
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warnings; enable TF32 where available
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args = parse_args()
    evaluate(args)