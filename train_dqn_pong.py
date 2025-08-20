"""
Script to train two DQN agents on Pong using Stable Baselines 3.
Trains both agents for 1,000,000 timesteps with a 16-frame stack and compares their performance.

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
import numpy as np
import time

# Training parameters - OPTIMIZED FOR EFFICIENCY
ENV_NAME = "ALE/Pong-v5"

# Adaptive training: start smaller, extend if needed
MIN_TIMESTEPS = 100_000      # Minimum for meaningful learning
MAX_TIMESTEPS = 400_000      # Maximum if needed
TARGET_PERFORMANCE = -12.0   # Stop early if achieved (Pong: -21 to +21, -12 is good)

FRAME_STACK_1 = 4   # Agent 1: 4-frame stack (optimal for Pong)
FRAME_STACK_2 = 8   # Agent 2: 8-frame stack (reduced from 16 for speed)
LOG_DIR_1 = "./dqn_pong_agent1_logs/"
LOG_DIR_2 = "./dqn_pong_agent2_logs/"
MODEL_SAVE_PATH_1 = "./dqn_pong_agent1.zip"
MODEL_SAVE_PATH_2 = "./dqn_pong_agent2.zip"

# Create environments for both agents with different frame stacks
def create_env(frame_stack):
    env = make_vec_env(
        ENV_NAME,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"render_mode": "rgb_array"}
    )
    return VecFrameStack(env, n_stack=frame_stack)

env1 = create_env(FRAME_STACK_1)  # 4-frame stack
env2 = create_env(FRAME_STACK_2)  # 16-frame stack

# Create two DQN models with different configurations for comparison
# Agent 1: 4-frame stack configuration - OPTIMIZED FOR SPEED
model1 = DQN(
    "CnnPolicy",
    env1,
    verbose=1,
    tensorboard_log=LOG_DIR_1,
    learning_rate=5e-4,        # Higher LR for faster learning
    buffer_size=100_000,       # Larger buffer for better sample efficiency
    learning_starts=10_000,    # Start learning after more experience
    batch_size=64,             # Larger batch for more stable gradients
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=2,          # More gradient steps per update
    target_update_interval=1_000,  # More frequent target updates
    exploration_fraction=0.05, # Shorter exploration phase
    exploration_final_eps=0.005,  # Lower final epsilon
    seed=42
)

# Agent 2: 8-frame stack configuration - MEMORY OPTIMIZED
model2 = DQN(
    "CnnPolicy",
    env2,
    verbose=1,
    tensorboard_log=LOG_DIR_2,
    learning_rate=3e-4,        # Balanced LR for 8-frame stack
    buffer_size=60_000,        # Reduced buffer due to memory constraints
    learning_starts=10_000,    # Same learning starts  
    batch_size=32,             # Smaller batch due to larger frame stack
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,          # Single gradient step due to memory
    target_update_interval=1_000,  # More frequent updates
    exploration_fraction=0.05, # Shorter exploration
    exploration_final_eps=0.005,  # Lower final epsilon
    seed=42
)

# Create evaluation environments and callbacks for both agents
eval_env1 = create_env(FRAME_STACK_1)  # 4-frame stack
eval_env2 = create_env(FRAME_STACK_2)  # 8-frame stack

class SmartEarlyStoppingCallback(EvalCallback):
    """Enhanced callback with early stopping based on performance"""
    
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq, 
                 n_eval_episodes, target_performance=-12.0, patience=5):
        super().__init__(eval_env, best_model_save_path, log_path, eval_freq, 
                        n_eval_episodes, deterministic=True, render=False)
        self.target_performance = target_performance
        self.patience = patience
        self.no_improvement_count = 0
        self.best_mean_reward = float('-inf')
        
    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        
        if self.last_mean_reward is not None:
            # Check if target performance reached
            if self.last_mean_reward >= self.target_performance:
                print(f"🎯 Target performance {self.target_performance} reached! "
                      f"Current: {self.last_mean_reward:.2f}")
                return False  # Stop training
            
            # Check for improvement
            if self.last_mean_reward > self.best_mean_reward + 1.0:  # 1.0 improvement threshold
                self.best_mean_reward = self.last_mean_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Early stop if no improvement for too long and past minimum steps
            if (self.no_improvement_count >= self.patience and 
                self.num_timesteps >= MIN_TIMESTEPS):
                print(f"🛑 Early stopping: No improvement for {self.patience} evaluations")
                return False
                
        return continue_training

eval_callback1 = SmartEarlyStoppingCallback(
    eval_env1,
    best_model_save_path=LOG_DIR_1,
    log_path=LOG_DIR_1,
    eval_freq=15_000,  # More frequent evaluation for early stopping
    n_eval_episodes=15,  # More episodes for reliable estimates
    target_performance=TARGET_PERFORMANCE,
    patience=3  # Stop after 3 evaluations without improvement
)

eval_callback2 = SmartEarlyStoppingCallback(
    eval_env2,
    best_model_save_path=LOG_DIR_2,
    log_path=LOG_DIR_2,
    eval_freq=15_000,
    n_eval_episodes=15,
    target_performance=TARGET_PERFORMANCE,
    patience=3
)

# Train both models with smart early stopping
print(f"🚀 SMART TRAINING: {ENV_NAME}")
print(f"Min timesteps: {MIN_TIMESTEPS:,}, Max timesteps: {MAX_TIMESTEPS:,}")
print(f"Target performance: {TARGET_PERFORMANCE}")
print(f"Agent 1: {FRAME_STACK_1}-frame stack (100k buffer, optimized)")
print(f"Agent 2: {FRAME_STACK_2}-frame stack (60k buffer, memory-optimized)")
print("Early stopping enabled - will stop when target reached or no improvement")
print("=" * 80)

# Train Agent 1
print("\n🤖 Training Agent 1 (4-frame stack)...")
start_time1 = time.time()
try:
    model1.learn(
        total_timesteps=MAX_TIMESTEPS,
        callback=eval_callback1,
        progress_bar=True
    )
except KeyboardInterrupt:
    print("Training interrupted by user")
train_time1 = time.time() - start_time1
actual_timesteps1 = model1.num_timesteps

# Train Agent 2
print(f"\n🤖 Training Agent 2 ({FRAME_STACK_2}-frame stack)...")
start_time2 = time.time()
try:
    model2.learn(
        total_timesteps=MAX_TIMESTEPS,
        callback=eval_callback2,
        progress_bar=True
    )
except KeyboardInterrupt:
    print("Training interrupted by user")
train_time2 = time.time() - start_time2
actual_timesteps2 = model2.num_timesteps

# Save both models
model1.save(MODEL_SAVE_PATH_1)
model2.save(MODEL_SAVE_PATH_2)
print(f"\nAgent 1 model saved to {MODEL_SAVE_PATH_1}")
print(f"Agent 2 model saved to {MODEL_SAVE_PATH_2}")

# Performance comparison
print("\n" + "=" * 80)
print("🏆 PERFORMANCE COMPARISON")
print("=" * 80)

def evaluate_agent(model, env, n_episodes=20):
    """Evaluate an agent over multiple episodes"""
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
        episode_rewards.append(episode_reward)
        
    return episode_rewards

# Create fresh evaluation environments
eval_final_env1 = create_env(FRAME_STACK_1)  # 4-frame stack
eval_final_env2 = create_env(FRAME_STACK_2)  # 16-frame stack

print("Evaluating Agent 1 (20 episodes)...")
rewards1 = evaluate_agent(model1, eval_final_env1, n_episodes=20)

print("Evaluating Agent 2 (20 episodes)...")
rewards2 = evaluate_agent(model2, eval_final_env2, n_episodes=20)

# Statistics
mean_reward1 = np.mean(rewards1)
std_reward1 = np.std(rewards1)
mean_reward2 = np.mean(rewards2)
std_reward2 = np.std(rewards2)

print(f"\n📊 SMART TRAINING RESULTS SUMMARY:")
print(f"Agent 1 ({FRAME_STACK_1}-frame stack):")
print(f"  - Training time: {train_time1:.1f} seconds ({train_time1/60:.1f} minutes)")
print(f"  - Timesteps used: {actual_timesteps1:,} / {MAX_TIMESTEPS:,}")
print(f"  - Efficiency: {actual_timesteps1/train_time1:.0f} steps/second")
print(f"  - Mean reward: {mean_reward1:.2f} ± {std_reward1:.2f}")
print(f"  - Best episode: {max(rewards1):.2f}")
print(f"  - Worst episode: {min(rewards1):.2f}")
print(f"  - Buffer size: 100,000")

print(f"\nAgent 2 ({FRAME_STACK_2}-frame stack):")
print(f"  - Training time: {train_time2:.1f} seconds ({train_time2/60:.1f} minutes)")
print(f"  - Timesteps used: {actual_timesteps2:,} / {MAX_TIMESTEPS:,}")
print(f"  - Efficiency: {actual_timesteps2/train_time2:.0f} steps/second")
print(f"  - Mean reward: {mean_reward2:.2f} ± {std_reward2:.2f}")
print(f"  - Best episode: {max(rewards2):.2f}")
print(f"  - Worst episode: {min(rewards2):.2f}")
print(f"  - Buffer size: 60,000")

# Determine winner
if mean_reward1 > mean_reward2:
    winner = f"Agent 1 ({FRAME_STACK_1}-frame stack)"
    improvement = ((mean_reward1 - mean_reward2) / abs(mean_reward2)) * 100
else:
    winner = f"Agent 2 ({FRAME_STACK_2}-frame stack)"
    improvement = ((mean_reward2 - mean_reward1) / abs(mean_reward1)) * 100

print(f"\n🎯 WINNER: {winner}")
print(f"Performance improvement: {improvement:.1f}%")
print(f"\n💡 INSIGHTS:")
print(f"This comparison shows whether more temporal information ({FRAME_STACK_2} frames)")
print(f"outweighs the benefits of a larger replay buffer ({FRAME_STACK_1} frames with 2x larger buffer).")

# Close environments
env1.close()
env2.close()
eval_final_env1.close()
eval_final_env2.close()
eval_env1.close()
eval_env2.close()