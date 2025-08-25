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
from sub_models.statemask import StateMaskGate


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


def eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent, eval_seed=42,
                  statemask: StateMaskGate = None):
    world_model.eval()
    agent.eval()
    
    # Set evaluation seed for reproducible episodes
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs)
    
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    
    # Properly seed the vector environment
    seeds = [eval_seed + i for i in range(num_envs)]
    current_obs, current_info = vec_env.reset(seed=seeds)  # Seed each environment
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    final_rewards = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    while True:
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                # Use seeded random for initial actions
                np.random.seed(eval_seed + len(final_rewards))
                action = vec_env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=True,
                    statemask=statemask
                )

        context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    if len(final_rewards) == num_episode:
                        print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(final_rewards)}" + colorama.Style.RESET_ALL)
                        return np.mean(final_rewards)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    parser.add_argument("-eval_seed", type=int, default=42, help="Seed for evaluation episodes")
    parser.add_argument("--statemask_eval", action="store_true", help="Apply saved StateMask during evaluation")
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    # print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=conf.BasicSettings.Seed)

    # build and load model/agent
    import train
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt/{args.run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    if not pathes:
        print(f"ERROR: No checkpoint files found in {root_path}/")
        print(f"Looking for pattern: {root_path}/world_model_*.pth")
        print("Available files in checkpoint directory:")
        if os.path.exists(root_path):
            for file in os.listdir(root_path):
                print(f"  - {file}")
        else:
            print(f"  Checkpoint directory {root_path} does not exist!")
        exit(1)
    
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[-1:]
    print(f"Found checkpoints for steps: {steps}")
    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # Optionally load StateMask for masked-policy evaluation
        statemask = None
        if args.statemask_eval:
            statemask_path = f"{root_path}/statemask_{step}.pth"
            if os.path.exists(statemask_path):
                feat_dim = 32*32 + conf.Models.WorldModel.TransformerHiddenDim
                hidden_dim = getattr(conf.Models.StateMask, 'HiddenDim', 128) if hasattr(conf.Models, 'StateMask') else 128
                statemask = StateMaskGate(feat_dim=feat_dim, hidden_dim=hidden_dim)
                statemask.load_state_dict(torch.load(statemask_path))
                device = next(agent.parameters()).device
                statemask = statemask.to(device)
                print(colorama.Fore.CYAN + f"Loaded StateMask from {statemask_path}" + colorama.Style.RESET_ALL)
            else:
                print(colorama.Fore.YELLOW + f"StateMask checkpoint not found for step {step}; proceeding without mask" + colorama.Style.RESET_ALL)
        # # eval
        episode_avg_return = eval_episodes(
            num_episode=20,
            env_name=args.env_name,
            num_envs=5,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            world_model=world_model,
            agent=agent,
            eval_seed=args.eval_seed,  # Use command line seed
            statemask=statemask
        )
        results.append([step, episode_avg_return])
    
    os.makedirs('eval_result', exist_ok=True)
    with open(f"eval_result/{args.run_name}.csv", "w") as fout:
        fout.write("step, episode_avg_return\n")
        for step, episode_avg_return in results:
            fout.write(f"{step},{episode_avg_return}\n")
