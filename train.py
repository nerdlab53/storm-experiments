import gymnasium
import argparse
from tensorboardX import SummaryWriter
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import colorama
import shutil
import os
from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss
from sub_models.world_models_adamae import WorldModel as AdaMAESTORM
from device_utils import move_to_device, print_device_info
import wandb


def build_single_env(env_name, image_size, seed):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda : build_single_env(env_name, image_size, seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger, current_step=0):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    # Convert observations from H W C to C H W format for the encoder
    obs = rearrange(obs, "B L H W C -> B L C H W")
    world_model.update(obs, action, reward, termination, logger=logger, current_step=current_step)


@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    # If we are logging video, also fetch ground-truth future segment for side-by-side comparison
    total_length = imagine_context_length + (imagine_batch_length if log_video else 0)
    sample_obs_full, sample_action_full, _, _ = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, max(imagine_context_length, total_length))
    # Split into context and future
    context_obs_np = sample_obs_full[:, :imagine_context_length]
    context_action_np = sample_action_full[:, :imagine_context_length]
    gt_future_obs_np = sample_obs_full[:, imagine_context_length:imagine_context_length+imagine_batch_length] if log_video else None
    gt_future_action_np = sample_action_full[:, imagine_context_length:imagine_context_length+imagine_batch_length] if log_video else None

    # Convert observations from H W C to C H W format for the encoder
    sample_obs = rearrange(context_obs_np, "B L H W C -> B L C H W")
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, context_action_np,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    # Side-by-side GT vs Predicted (teacher-forced rollout using GT actions)
    if log_video:
        try:
            B = context_obs_np.shape[0]
            # Prepare tensors
            context_obs = move_to_device(torch.tensor(context_obs_np))  # B, Lc, H, W, C
            context_obs = rearrange(context_obs, "B L H W C -> B L C H W") / 255.0
            context_action = move_to_device(torch.tensor(context_action_np))  # B, Lc
            gt_future_obs = None
            gt_future_action = None
            if gt_future_obs_np is not None:
                gt_future_obs = move_to_device(torch.tensor(gt_future_obs_np))  # B, Lf, H, W, C
                gt_future_obs = rearrange(gt_future_obs, "B L H W C -> B L C H W") / 255.0
            if gt_future_action_np is not None:
                gt_future_action = move_to_device(torch.tensor(gt_future_action_np))  # B, Lf

            if gt_future_obs is not None and gt_future_action is not None and gt_future_obs.shape[1] > 0:
                # Reset KV cache and advance through context using GT actions
                world_model.storm_transformer.reset_kv_cache_list(B, dtype=world_model.tensor_dtype)
                context_latent = world_model.encode_obs(context_obs)
                for i in range(context_action.shape[1]):
                    _, _, _, last_latent, last_dist_feat = world_model.predict_next(
                        context_latent[:, i:i+1], context_action[:, i:i+1], log_video=True)
                # Teacher-forced rollout following GT actions and decode predicted frames
                pred_list = []
                gt_list = []
                for t in range(gt_future_action.shape[1]):
                    last_obs_hat, _, _, last_latent, last_dist_feat = world_model.predict_next(
                        last_latent, gt_future_action[:, t:t+1], log_video=True)
                    pred_list.append(last_obs_hat)
                    gt_list.append(gt_future_obs[:, t:t+1])

                # Stack time dimension
                pred_video = torch.cat(pred_list, dim=1)  # B, T, C, H, W
                gt_video = torch.cat(gt_list, dim=1)      # B, T, C, H, W

                # Uniformly sample up to 16 sequences across batch like existing logging
                stride = max(1, B // 16)
                pred_video = pred_video[::stride]
                gt_video = gt_video[::stride]

                # Side-by-side along width: (B, T, C, H, 2W)
                side_by_side = torch.cat([gt_video, pred_video], dim=4).clamp(0, 1)
                logger.log("Imagine/gt_vs_pred_video", side_by_side.cpu().float().detach().numpy())
        except Exception:
            # Never let logging break training
            pass

    return latent, action, None, None, reward_hat, termination_hat


def joint_train_world_model_agent(env_name, max_steps, num_envs, image_size,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every_steps, seed, logger):
    # create ckpt dir
    os.makedirs(f"ckpt/{args.n}", exist_ok=True)

    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    # sample and train
    for total_steps in tqdm(range(max_steps//num_envs)):
        # sync global step for wandb so scalar logs align
        logger.set_global_step(total_steps * num_envs)
        # sample part >>>
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    # get posterior logits
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = move_to_device(torch.Tensor(model_context_action))
                    # prior_flattened_sample represents what the model thinks the current state looks like
                    # last_dist_feat -> sequential info from the transformer containing imp information such as dynamics etc. basically which led to the logits
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )        
            context_obs.append(rearrange(move_to_device(torch.Tensor(current_obs)), "B H W C -> B 1 C H W")/255)
            context_action.append(action)
        else:
            action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(action)
        replay_buffer.append(current_obs, action, reward, np.logical_or(done, info["life_loss"]))

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    # Log episodic metrics at episode boundaries; keep TB step per-tag and WandB step via global_step
                    logger.log(f"sample/{env_name}_reward", float(sum_reward[i]))
                    logger.log(f"sample/{env_name}_episode_steps", int(current_info["episode_frame_number"][i]//4))  # frameskip=4
                    logger.log("replay_buffer/length", len(replay_buffer))
                    sum_reward[i] = 0

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part

        # train world model part >>>
        if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                logger=logger,
                current_step=total_steps * num_envs
            )
        # <<< train world model part

        # train agent part >>>
        if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
            else:
                log_video = False

            imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=imagine_batch_size,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                log_video=log_video,
                logger=logger
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger
            )
        # <<< train agent part

        # save model per episode
        if total_steps % (save_every_steps//num_envs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"ckpt/{args.n}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"ckpt/{args.n}/agent_{total_steps}.pth")
            

def build_world_model(conf, action_dim, use_adamae=False):
    model_cls = AdaMAESTORM if use_adamae else WorldModel
    print(colorama.Fore.CYAN + f"Using {'AdaMAEStorm' if use_adamae else 'STORM'} world model" + colorama.Style.RESET_ALL)
    
    base_args = {
        'in_channels': conf.Models.WorldModel.InChannels,
        'action_dim': action_dim,
        'transformer_max_length': conf.Models.WorldModel.TransformerMaxLength,
        'transformer_hidden_dim': conf.Models.WorldModel.TransformerHiddenDim,
        'transformer_num_layers': conf.Models.WorldModel.TransformerNumLayers,
        'transformer_num_heads': conf.Models.WorldModel.TransformerNumHeads,
        'use_progressive_masking': getattr(conf.Models.WorldModel, 'UseProgressiveMasking', True),
        'use_progressive_in_kv': getattr(conf.Models.WorldModel, 'UseProgressiveInKVCache', False),
        'use_mild_decay_in_kv': getattr(conf.Models.WorldModel, 'UseMildDecayInKV', False),
        'fixed_mask_percent': getattr(conf.Models.WorldModel, 'FixedMaskPercent', 0.0),
        'use_random_mask': getattr(conf.Models.WorldModel, 'UseRandomMask', False),
        'use_soft_penalty': getattr(conf.Models.WorldModel, 'UseSoftPenalty', True)
    }
    
    world_model = move_to_device(model_cls(**base_args))
    
    # Configure AdaMAE-specific mask schedule if using AdaMAE
    if use_adamae:
        world_model.use_mask_schedule = getattr(conf.Models.WorldModel, 'UseMaskSchedule', False)
        world_model.mask_ratio = getattr(conf.Models.WorldModel, 'MaskRatio', 0.50)
        world_model.mask_ratio_start = getattr(conf.Models.WorldModel, 'MaskRatioStart', 0.25)
        world_model.mask_ratio_end = getattr(conf.Models.WorldModel, 'MaskRatioEnd', 0.75)
        world_model.mask_warmup_steps = getattr(conf.Models.WorldModel, 'MaskWarmupSteps', 20000)
        
        if world_model.use_mask_schedule:
            print(colorama.Fore.MAGENTA + f"AdaMAE adaptive masking: {world_model.mask_ratio_start:.0%} → {world_model.mask_ratio_end:.0%} over {world_model.mask_warmup_steps} steps" + colorama.Style.RESET_ALL)
        else:
            print(colorama.Fore.MAGENTA + f"AdaMAE fixed masking: {world_model.mask_ratio:.0%}" + colorama.Style.RESET_ALL)
    
    return world_model


def build_agent(conf, action_dim):
    return move_to_device(agents.ActorCriticAgent(
        feat_dim=32*32+conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
    ))


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-trajectory_path", type=str, required=True)
    parser.add_argument("--use_adamae", action="store_true", help="Use AdaMAEStorm world model")
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # Print device information
    print_device_info()

    # set seed
    seed_np_torch(seed=args.seed)
    # Initialize Weights & Biases (optional if WANDB_DISABLED=1)
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    try:
        wandb.init(project=os.environ.get("WANDB_PROJECT", "STORM-exps"),
                   name=args.n,
                   config={},
                   mode=wandb_mode)
    except Exception:
        pass
    # tensorboard writer (kept as is)
    logger = Logger(path=f"runs/{args.n}")
    # copy config file
    shutil.copy(args.config_path, f"runs/{args.n}/config.yaml")

    # Log config and args to wandb if active
    if wandb.run is not None:
        try:
            # yacs CfgNode -> dict
            wandb.config.update({"args": vars(args)}, allow_val_change=True)
            # Best-effort deep conversion of CfgNode
            def cfg_to_dict(cfg):
                try:
                    return cfg.to_dict()
                except Exception:
                    return {}
            wandb.config.update({"config": cfg_to_dict(conf)}, allow_val_change=True)
        except Exception:
            pass

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize, seed=0)
        action_dim = dummy_env.action_space.n

        # build world model and agent
        world_model = build_world_model(conf, action_dim, use_adamae=args.use_adamae)
        agent = build_agent(conf, action_dim)

        # build replay buffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU
        )

        # judge whether to load demonstration trajectory
        if conf.JointTrainAgent.UseDemonstration:
            print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {args.trajectory_path}" + colorama.Style.RESET_ALL)
            replay_buffer.load_trajectory(path=args.trajectory_path)

        # train
        joint_train_world_model_agent(
            env_name=args.env_name,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            batch_length=conf.JointTrainAgent.BatchLength,
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            seed=args.seed,
            logger=logger,
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")