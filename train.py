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

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss
from sub_models.novelty_detector import WorldModelNoveltyWrapper
from novelty_injector import NoveltyEnvironmentWrapper, NoveltyInjector, PREDEFINED_NOVELTIES
from device_utils import get_device, move_to_device, print_device_info, DEVICE


def build_single_env(env_name, image_size, seed):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed, novelty_config=None):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size, novelty_config):
        def make_env():
            base_env = build_single_env(env_name, image_size, seed)
            if novelty_config and novelty_config.get('Enabled', False):
                env = NoveltyEnvironmentWrapper(base_env)
                # Configure novelty injection
                env.configure_novelty(
                    novelty_config['NoveltyType'],
                    novelty_config['NoveltyParams'],
                    novelty_config['NoveltyStartStep']
                )
                return env
            return base_env
        return make_env
    
    env_fns = [lambda_generator(env_name, image_size, novelty_config) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    # Convert observations from H W C to C H W format for the encoder
    obs = rearrange(obs, "B L H W C -> B L C H W")
    world_model.update(obs, action, reward, termination, logger=logger)


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

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    # Convert observations from H W C to C H W format for the encoder
    sample_obs = rearrange(sample_obs, "B L H W C -> B L C H W")
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat


def joint_train_world_model_agent(env_name, max_steps, num_envs, image_size,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every_steps, seed, logger, novelty_config=None):
    # create ckpt dir
    os.makedirs(f"ckpt/{args.n}", exist_ok=True)

    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed, novelty_config=novelty_config)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    
    # Check if novelty detection is enabled
    novelty_detection_enabled = hasattr(world_model, 'novelty_detector')
    if novelty_detection_enabled:
        print(colorama.Fore.CYAN + "Novelty detection enabled during training" + colorama.Style.RESET_ALL)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample and train
    for total_steps in tqdm(range(max_steps//num_envs)):
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
                    
                    # check if novelty detection enabled and then perform
                    if novelty_detection_enabled and len(context_obs) > 0:
                        try:
                            is_novelty, detection_info = world_model.detect_novelty_step(
                                context_obs[-1],  # the current obeservation
                                move_to_device(torch.tensor([action[0]])),  # current action (first env)
                                last_dist_feat,  # latent context from transformer
                                total_steps * num_envs
                            )
                            
                            if is_novelty:
                                # print(colorama.Fore.RED + f"NOVELTY DETECTED at step {total_steps * num_envs}!" + colorama.Style.RESET_ALL), if needed, otherwise we'll check the logs only
                                logger.log("novelty_detection/detection_flag", 1.0)
                            else:
                                logger.log("novelty_detection/detection_flag", 0.0)
                            
                            # Log detection metrics
                            logger.log("novelty_detection/kl_difference", detection_info.get('kl_difference', 0.0))
                            logger.log("novelty_detection/expected_info_gain", detection_info.get('expected_info_gain', 0.0))
                            logger.log("novelty_detection/threshold", detection_info.get('threshold', 0.0))
                            logger.log("novelty_detection/detection_rate", detection_info.get('detection_rate', 0.0))
                        except Exception as e:
                            print(f"Novelty detection error: {e}")

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
                    logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    logger.log(f"sample/{env_name}_episode_steps", current_info["episode_frame_number"][i]//4)  # framskip=4
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
                logger=logger
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
            
            # Save novelty detection logs if enabled
            if novelty_detection_enabled:
                log_path = f"ckpt/{args.n}/novelty_detection_{total_steps}.json"
                world_model.save_detection_log(log_path)
                print(colorama.Fore.CYAN + f"Saved novelty detection log to {log_path}" + colorama.Style.RESET_ALL)


def build_world_model(conf, action_dim):
    world_model = move_to_device(WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
        use_progressive_masking=getattr(conf.Models.WorldModel, 'UseProgressiveMasking', True),
        use_progressive_in_kv=getattr(conf.Models.WorldModel, 'UseProgressiveInKVCache', False),
        use_mild_decay_in_kv=getattr(conf.Models.WorldModel, 'UseMildDecayInKV', False),
        fixed_mask_percent=getattr(conf.Models.WorldModel, 'FixedMaskPercent', 0.0),
        fixed_mask_percents=getattr(conf.Models.WorldModel, 'FixedMaskPercents', None),
        use_random_mask=getattr(conf.Models.WorldModel, 'UseRandomMask', False),
        use_soft_penalty=getattr(conf.Models.WorldModel, 'UseSoftPenalty', True)
    ))
    
    # Add novelty detection if enabled
    if hasattr(conf.Models, 'NoveltyDetection') and getattr(conf.Models.NoveltyDetection, 'Enabled', False):
        print(colorama.Fore.CYAN + "Enabling novelty detection..." + colorama.Style.RESET_ALL)
        world_model = WorldModelNoveltyWrapper(
            world_model,
            history_length=getattr(conf.Models.NoveltyDetection, 'HistoryLength', 100),
            detection_threshold_percentile=getattr(conf.Models.NoveltyDetection, 'DetectionThresholdPercentile', 95.0),
            min_samples_for_detection=getattr(conf.Models.NoveltyDetection, 'MinSamplesForDetection', 50),
            enable_adaptive_threshold=getattr(conf.Models.NoveltyDetection, 'EnableAdaptiveThreshold', True),
            eig_threshold=getattr(conf.Models.NoveltyDetection, 'EIGThreshold', 0.0),
            use_eig_primary=getattr(conf.Models.NoveltyDetection, 'UseEIGPrimary', True)
        )
        
        # Create detection log directory
        log_path = getattr(conf.Models.NoveltyDetection, 'DetectionLogPath', 'detection_logs/')
        os.makedirs(log_path, exist_ok=True)
    
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
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # Print device information
    print_device_info()

    # set seed
    seed_np_torch(seed=args.seed)
    # tensorboard writer
    logger = Logger(path=f"runs/{args.n}")
    # copy config file
    shutil.copy(args.config_path, f"runs/{args.n}/config.yaml")

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize, seed=0)
        action_dim = dummy_env.action_space.n

        # build world model and agent
        world_model = build_world_model(conf, action_dim)
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

        # Get novelty testing configuration if available
        novelty_config = None
        if hasattr(conf, 'NoveltyTesting'):
            # Convert YACS NoveltyParams to dictionary
            novelty_params_dict = {}
            if hasattr(conf.NoveltyTesting, 'NoveltyParams'):
                for key in conf.NoveltyTesting.NoveltyParams:
                    novelty_params_dict[key] = getattr(conf.NoveltyTesting.NoveltyParams, key)
            
            novelty_config = {
                'Enabled': getattr(conf.NoveltyTesting, 'Enabled', False),
                'NoveltyType': getattr(conf.NoveltyTesting, 'NoveltyType', 'visual_noise'),
                'NoveltyParams': novelty_params_dict,
                'NoveltyStartStep': getattr(conf.NoveltyTesting, 'NoveltyStartStep', 100)
            }
            if novelty_config['Enabled']:
                print(colorama.Fore.MAGENTA + f"Novelty testing enabled: {novelty_config['NoveltyType']} starting at step {novelty_config['NoveltyStartStep']}" + colorama.Style.RESET_ALL)

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
            novelty_config=novelty_config
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")
