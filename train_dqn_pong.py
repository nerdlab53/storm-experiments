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

# Training parameters
ENV_NAME = "ALE/Pong-v5"
TOTAL_TIMESTEPS = 500_000  # Further reduced for memory constraints
FRAME_STACK_1 = 4   # Agent 1: 4-frame stack
FRAME_STACK_2 = 8   # Agent 2: 8-frame stack 
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
# Agent 1: 4-frame stack configuration
model1 = DQN(
    "CnnPolicy",
    env1,
    verbose=1,
    tensorboard_log=LOG_DIR_1,
    learning_rate=1e-4,
    buffer_size=30_000,  
    learning_starts=3_000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=3_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    seed=42
)

# Agent 2: 8-frame stack configuration
model2 = DQN(
    "CnnPolicy",
    env2,
    verbose=1,
    tensorboard_log=LOG_DIR_2,
    learning_rate=1e-4,  # Same learning rate for fair comparison
    buffer_size=30_000,  # Smaller buffer due to 8-frame stack memory usage
    learning_starts=3_000,  # Same learning starts
    batch_size=32,  # Same batch size
    tau=1.0,
    gamma=0.99,  # Same gamma for fair comparison
    train_freq=4,
    gradient_steps=1,
    target_update_interval=3_000,  # Same target update interval
    exploration_fraction=0.1,  # Same exploration
    exploration_final_eps=0.01,  # Same final epsilon
    seed=42  # Same seed for fair comparison
)

# Create evaluation environments and callbacks for both agents
eval_env1 = create_env(FRAME_STACK_1)  # 4-frame stack
eval_env2 = create_env(FRAME_STACK_2)  # 8-frame stack

eval_callback1 = EvalCallback(
    eval_env1,
    best_model_save_path=LOG_DIR_1,
    log_path=LOG_DIR_1,
    eval_freq=20_000,  # Evaluate every 20k steps (adjusted for shorter training)
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

eval_callback2 = EvalCallback(
    eval_env2,
    best_model_save_path=LOG_DIR_2,
    log_path=LOG_DIR_2,
    eval_freq=20_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

# Train both models in chunks of 100,000 steps
CHUNK_SIZE = 100_000
num_chunks = TOTAL_TIMESTEPS // CHUNK_SIZE

for chunk in range(num_chunks):
    print(f"\n🔄 Training chunk {chunk + 1}/{num_chunks} ({CHUNK_SIZE} steps)...")
    
    # Train Agent 2 chunk
    print("  Training Agent 2...")
    model2.learn(
        total_timesteps=CHUNK_SIZE,
        callback=eval_callback2,
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    # Train Agent 1 chunk
    print("  Training Agent 1...")
    model1.learn(
        total_timesteps=CHUNK_SIZE,
        callback=eval_callback1,
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    # Evaluate and save after chunk
    current_step = (chunk + 1) * CHUNK_SIZE
    print(f"\n📊 Evaluating after {current_step} steps...")
    
    rewards1 = evaluate_agent(model1, eval_final_env1, n_episodes=20)
    rewards2 = evaluate_agent(model2, eval_final_env2, n_episodes=20)
    
    mean_reward1 = np.mean(rewards1)
    std_reward1 = np.std(rewards1)
    mean_reward2 = np.mean(rewards2)
    std_reward2 = np.std(rewards2)
    
    if mean_reward1 > mean_reward2:
        winner = f"Agent 1 ({FRAME_STACK_1}-frame stack)"
        improvement = ((mean_reward1 - mean_reward2) / abs(mean_reward2)) * 100 if mean_reward2 != 0 else float('inf')
    else:
        winner = f"Agent 2 ({FRAME_STACK_2}-frame stack)"
        improvement = ((mean_reward2 - mean_reward1) / abs(mean_reward1)) * 100 if mean_reward1 != 0 else float('inf')
    
    # Save to chunk-specific file
    chunk_filename = f'dqn_pong_eval_results_step_{current_step}.txt'
    with open(chunk_filename, 'w') as f:
        f.write(f"DQN Pong Evaluation Results - Step {current_step}\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Agent 1 ({FRAME_STACK_1}-frame stack):\n")
        f.write(f"  - Mean reward: {mean_reward1:.2f} ± {std_reward1:.2f}\n")
        f.write(f"  - Best episode: {max(rewards1):.2f}\n")
        f.write(f"  - Worst episode: {min(rewards1):.2f}\n\n")
        
        f.write(f"Agent 2 ({FRAME_STACK_2}-frame stack):\n")
        f.write(f"  - Mean reward: {mean_reward2:.2f} ± {std_reward2:.2f}\n")
        f.write(f"  - Best episode: {max(rewards2):.2f}\n")
        f.write(f"  - Worst episode: {min(rewards2):.2f}\n\n")
        
        f.write(f"WINNER at this step: {winner}\n")
        f.write(f"Performance improvement: {improvement:.1f}%\n")
    
    print(f"📝 Intermediate results saved to {chunk_filename}")

# Final full evaluation (as before)
print("\n🏆 FINAL PERFORMANCE COMPARISON")
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
eval_final_env2 = create_env(FRAME_STACK_2)  # 8-frame stack

print("Evaluating Agent 1 (20 episodes)...")
rewards1 = evaluate_agent(model1, eval_final_env1, n_episodes=20)

print("Evaluating Agent 2 (20 episodes)...")
rewards2 = evaluate_agent(model2, eval_final_env2, n_episodes=20)

# Statistics
mean_reward1 = np.mean(rewards1)
std_reward1 = np.std(rewards1)
mean_reward2 = np.mean(rewards2)
std_reward2 = np.std(rewards2)

print(f"\n📊 RESULTS SUMMARY:")
print(f"Agent 1 ({FRAME_STACK_1}-frame stack):")
print(f"  - Training time: {train_time1:.1f} seconds")
print(f"  - Mean reward: {mean_reward1:.2f} ± {std_reward1:.2f}")
print(f"  - Best episode: {max(rewards1):.2f}")
print(f"  - Worst episode: {min(rewards1):.2f}")
print(f"  - Buffer size: 30,000")

print(f"\nAgent 2 ({FRAME_STACK_2}-frame stack):")
print(f"  - Training time: {train_time2:.1f} seconds")
print(f"  - Mean reward: {mean_reward2:.2f} ± {std_reward2:.2f}")
print(f"  - Best episode: {max(rewards2):.2f}")
print(f"  - Worst episode: {min(rewards2):.2f}")
print(f"  - Buffer size: 15,000")

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

# Save evaluation results to TXT file
with open('dqn_pong_eval_results.txt', 'w') as f:
    f.write("DQN Pong Evaluation Results\n")
    f.write("=" * 40 + "\n\n")
    
    f.write(f"Agent 1 ({FRAME_STACK_1}-frame stack):\n")
    f.write(f"  - Training time: {train_time1:.1f} seconds\n")
    f.write(f"  - Mean reward: {mean_reward1:.2f} ± {std_reward1:.2f}\n")
    f.write(f"  - Best episode: {max(rewards1):.2f}\n")
    f.write(f"  - Worst episode: {min(rewards1):.2f}\n")
    f.write(f"  - Buffer size: 30,000\n\n")
    
    f.write(f"Agent 2 ({FRAME_STACK_2}-frame stack):\n")
    f.write(f"  - Training time: {train_time2:.1f} seconds\n")
    f.write(f"  - Mean reward: {mean_reward2:.2f} ± {std_reward2:.2f}\n")
    f.write(f"  - Best episode: {max(rewards2):.2f}\n")
    f.write(f"  - Worst episode: {min(rewards2):.2f}\n")
    f.write(f"  - Buffer size: 15,000\n\n")
    
    f.write(f"WINNER: {winner}\n")
    f.write(f"Performance improvement: {improvement:.1f}%\n")

print("\n📝 Evaluation results saved to dqn_pong_eval_results.txt")

# Close environments
env1.close()
env2.close()
eval_final_env1.close()
eval_final_env2.close()
eval_env1.close()
eval_env2.close()