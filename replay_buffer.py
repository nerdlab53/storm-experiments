import torch
import numpy as np
import random
import pickle
from device_utils import DEVICE, move_to_device

class ReplayBuffer:
    def __init__(self, obs_shape, num_envs, max_length, warmup_length, store_on_gpu=True):
        self.obs_shape = obs_shape
        self.num_envs = num_envs
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.store_on_gpu = store_on_gpu

        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.uint8, device=DEVICE, requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=DEVICE, requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=DEVICE, requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=DEVICE, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)

        # demonstration_buffer
        self.demonstration_obs = []
        self.demonstration_action = []
        self.demonstration_reward = []
        self.demonstration_termination = []
        self.size = 0
        self.warmup = False

    def ready(self):
        return self.warmup

    def append(self, obs, action, reward, termination):
        if self.store_on_gpu:
            self.obs_buffer[self.size] = torch.from_numpy(obs).to(DEVICE)
            self.action_buffer[self.size] = torch.from_numpy(action).to(DEVICE)
            self.reward_buffer[self.size] = torch.from_numpy(reward).to(DEVICE)
            self.termination_buffer[self.size] = torch.from_numpy(termination).to(DEVICE)
        else:
            self.obs_buffer[self.size] = obs
            self.action_buffer[self.size] = action
            self.reward_buffer[self.size] = reward
            self.termination_buffer[self.size] = termination

        self.size = (self.size + 1) % (self.max_length//self.num_envs)
        if len(self) >= self.warmup_length:
            self.warmup = True

    def sample_external(self, batch_size, batch_length, to_device=None):
        if to_device is None:
            to_device = DEVICE
            
        obs = random.choices(self.demonstration_obs, k=batch_size)
        action = random.choices(self.demonstration_action, k=batch_size)
        reward = random.choices(self.demonstration_reward, k=batch_size)
        termination = random.choices(self.demonstration_termination, k=batch_size)

        obs = [np.expand_dims(x, axis=0) for x in obs]
        action = [np.expand_dims(x, axis=0) for x in action]
        reward = [np.expand_dims(x, axis=0) for x in reward]
        termination = [np.expand_dims(x, axis=0) for x in termination]

        obs = torch.from_numpy(np.concatenate(obs, axis=0)).float()
        obs = move_to_device(obs, to_device) / 255
        action = move_to_device(torch.from_numpy(np.concatenate(action, axis=0)), to_device)
        reward = move_to_device(torch.from_numpy(np.concatenate(reward, axis=0)), to_device)
        termination = move_to_device(torch.from_numpy(np.concatenate(termination, axis=0)), to_device)

        # Trim to batch_length
        obs = obs[:, :batch_length]
        action = action[:, :batch_length]
        reward = reward[:, :batch_length]
        termination = termination[:, :batch_length]

        return obs, action, reward, termination

    def sample(self, batch_size, external_batch_size, batch_length, to_device=None):
        if to_device is None:
            to_device = DEVICE
            
        # Sample from the replay buffer
        starts = torch.randint(0, len(self)-batch_length+1, (batch_size,), dtype=torch.long)
        
        if self.store_on_gpu:
            obs = torch.stack([self.obs_buffer[start:start+batch_length].flatten(0, 1) for start in starts])
            action = torch.stack([self.action_buffer[start:start+batch_length].flatten(0, 1) for start in starts])
            reward = torch.stack([self.reward_buffer[start:start+batch_length].flatten(0, 1) for start in starts])
            termination = torch.stack([self.termination_buffer[start:start+batch_length].flatten(0, 1) for start in starts])
            obs = obs.float() / 255
        else:
            obs = []
            action = []
            reward = []
            termination = []
            for start in starts:
                obs.append(self.obs_buffer[start:start+batch_length].flatten(0, 1))
                action.append(self.action_buffer[start:start+batch_length].flatten(0, 1))
                reward.append(self.reward_buffer[start:start+batch_length].flatten(0, 1))
                termination.append(self.termination_buffer[start:start+batch_length].flatten(0, 1))

            obs = torch.from_numpy(np.concatenate(obs, axis=0)).float()
            obs = move_to_device(obs, to_device) / 255
            action = move_to_device(torch.from_numpy(np.concatenate(action, axis=0)), to_device)
            reward = move_to_device(torch.from_numpy(np.concatenate(reward, axis=0)), to_device)
            termination = move_to_device(torch.from_numpy(np.concatenate(termination, axis=0)), to_device)

        # Sample from demonstration data if available and requested
        if external_batch_size > 0 and len(self.demonstration_obs) > 0:
            external_obs, external_action, external_reward, external_termination = self.sample_external(external_batch_size, batch_length, to_device)
            
            obs = torch.cat([obs, external_obs], dim=0)
            action = torch.cat([action, external_action], dim=0)
            reward = torch.cat([reward, external_reward], dim=0)
            termination = torch.cat([termination, external_termination], dim=0)

        return obs, action, reward, termination

    def load_trajectory(self, path):
        print(f"Loading demonstration trajectory from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.demonstration_obs = data["obs"]
        self.demonstration_action = data["action"]
        self.demonstration_reward = data["reward"]
        self.demonstration_termination = data["termination"]

    def __len__(self):
        if self.size == 0:
            return 0
        return min(self.size, self.max_length//self.num_envs)
