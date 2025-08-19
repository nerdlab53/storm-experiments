"""
Script to train a DQN agent on Pong using Stable Baselines 3.
Trains for 100,000 timesteps with a 16-frame stack.

Requirements:
- pip install stable-baselines3[extra] gymnasium[atari] gymnasium[accept-rom-license]

Usage:
python train_dqn_pong.py
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

# Training parameters
ENV_NAME = "ALE/Pong-v5"
TOTAL_TIMESTEPS = 100_000
FRAME_STACK = 16
LOG_DIR = "./dqn_pong_logs/"
MODEL_SAVE_PATH = "./dqn_pong_model.zip"

# Create the environment
# Use make_vec_env for easy wrapping (even for single env)
env = make_vec_env(
    ENV_NAME,
    n_envs=1,
    vec_env_cls=DummyVecEnv,
    env_kwargs={"render_mode": "rgb_array"}
)

# Apply frame stacking (16 frames)
env = VecFrameStack(env, n_stack=FRAME_STACK)

# Create the DQN model with CNN policy for Atari
model = DQN(
    "CnnPolicy",  # CNN policy for image-based environments
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=1e-4,  # Common value for Atari
    buffer_size=100_000,  # Replay buffer size
    learning_starts=10_000,  # Start learning after initial exploration
    batch_size=32,
    tau=1.0,  # Soft update
    gamma=0.99,  # Discount factor
    train_freq=4,  # Train every 4 steps
    gradient_steps=1,
    target_update_interval=10_000,  # Update target network periodically
    exploration_fraction=0.1,  # Fraction of training for exploration decay
    exploration_final_eps=0.01,  # Final epsilon for epsilon-greedy
    seed=42  # For reproducibility
)

# Optional: Evaluation callback to monitor progress
eval_env = make_vec_env(ENV_NAME, n_envs=1)
eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=LOG_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,  # Evaluate every 10k steps
    n_eval_episodes=10,  # Run 10 episodes for evaluation
    deterministic=True,
    render=False
)

# Train the model
print(f"Training DQN on {ENV_NAME} for {TOTAL_TIMESTEPS} timesteps with {FRAME_STACK}-frame stack...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    progress_bar=True
)

# Save the final model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Quick evaluation
print("Evaluating trained model...")
obs = env.reset()
total_reward = 0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
print(f"Evaluation reward: {total_reward}")

env.close()