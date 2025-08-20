"""
Smart training manager that automatically balances training time vs performance
Uses early stopping, performance monitoring, and adaptive scheduling
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class AdaptiveTrainingManager:
    """Manages training with intelligent early stopping and scheduling"""
    
    def __init__(self, 
                 min_timesteps: int = 50_000,
                 max_timesteps: int = 500_000,
                 target_performance: float = -10.0,
                 patience_episodes: int = 100,
                 improvement_threshold: float = 1.0):
        """
        Args:
            min_timesteps: Minimum training before considering early stop
            max_timesteps: Maximum training timesteps
            target_performance: Stop if this performance is reached
            patience_episodes: Episodes to wait without improvement
            improvement_threshold: Minimum improvement to continue training
        """
        self.min_timesteps = min_timesteps
        self.max_timesteps = max_timesteps
        self.target_performance = target_performance
        self.patience_episodes = patience_episodes
        self.improvement_threshold = improvement_threshold
        
        self.performance_history = []
        self.timestep_history = []
        self.training_times = []
        self.best_performance = float('-inf')
        self.episodes_without_improvement = 0
        
    def should_stop_training(self, current_performance: float, 
                           current_timesteps: int) -> Tuple[bool, str]:
        """Determine if training should stop based on performance"""
        
        # Always train for minimum timesteps
        if current_timesteps < self.min_timesteps:
            return False, "Below minimum timesteps"
            
        # Stop if target performance reached
        if current_performance >= self.target_performance:
            return True, f"Target performance {self.target_performance} reached"
            
        # Stop if maximum timesteps reached
        if current_timesteps >= self.max_timesteps:
            return True, "Maximum timesteps reached"
            
        # Check for improvement
        if current_performance > self.best_performance + self.improvement_threshold:
            self.best_performance = current_performance
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
            
        # Stop if no improvement for too long
        if self.episodes_without_improvement >= self.patience_episodes:
            return True, f"No improvement for {self.patience_episodes} episodes"
            
        return False, "Continue training"
    
    def log_performance(self, performance: float, timesteps: int, training_time: float):
        """Log performance metrics"""
        self.performance_history.append(performance)
        self.timestep_history.append(timesteps)
        self.training_times.append(training_time)
        
    def get_efficiency_score(self) -> float:
        """Calculate training efficiency (performance / time)"""
        if not self.performance_history:
            return 0.0
        return self.performance_history[-1] / sum(self.training_times)
    
    def plot_training_progress(self, save_path: str = "training_progress.png"):
        """Plot training progress over time"""
        if len(self.performance_history) < 2:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Performance over timesteps
        ax1.plot(self.timestep_history, self.performance_history, 'b-', linewidth=2)
        ax1.axhline(y=self.target_performance, color='r', linestyle='--', 
                   label=f'Target: {self.target_performance}')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Performance')
        ax1.set_title('Training Performance Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Training time efficiency
        efficiency_scores = [p / sum(self.training_times[:i+1]) 
                           for i, p in enumerate(self.performance_history)]
        ax2.plot(self.timestep_history, efficiency_scores, 'g-', linewidth=2)
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Efficiency (Performance / Time)')
        ax2.set_title('Training Efficiency Over Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training progress saved to {save_path}")

class SmartEvalCallback(BaseCallback):
    """Custom callback for smart evaluation and early stopping"""
    
    def __init__(self, eval_env, training_manager: AdaptiveTrainingManager,
                 eval_freq: int = 10_000, n_eval_episodes: int = 10):
        super().__init__()
        self.eval_env = eval_env
        self.training_manager = training_manager
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        """Called after each environment step"""
        
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current performance
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                    
                episode_rewards.append(episode_reward)
            
            mean_performance = np.mean(episode_rewards)
            current_time = time.time() - self.start_time
            
            # Log performance
            self.training_manager.log_performance(
                mean_performance, self.n_calls, current_time
            )
            
            # Check if should stop
            should_stop, reason = self.training_manager.should_stop_training(
                mean_performance, self.n_calls
            )
            
            print(f"Step {self.n_calls}: Performance = {mean_performance:.2f}, "
                  f"Time = {current_time:.1f}s, Reason: {reason}")
            
            if should_stop:
                print(f"🛑 Early stopping: {reason}")
                return False  # Stop training
                
        return True  # Continue training

def create_optimized_dqn_config(env_name: str) -> Dict:
    """Create optimized DQN configuration for specific environment"""
    
    base_config = {
        "learning_rate": 2.5e-4,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
    }
    
    # Environment-specific optimizations
    if "Pong" in env_name:
        base_config.update({
            "learning_rate": 5e-4,  # Higher LR for Pong
            "buffer_size": 50_000,  # Smaller buffer sufficient
            "target_update_interval": 500,  # More frequent updates
        })
    elif "Breakout" in env_name:
        base_config.update({
            "learning_rate": 1e-4,  # More conservative for Breakout
            "buffer_size": 100_000,
            "exploration_fraction": 0.2,  # More exploration needed
        })
    elif "MsPacman" in env_name:
        base_config.update({
            "learning_rate": 2e-4,
            "buffer_size": 80_000,
            "exploration_fraction": 0.15,
        })
    
    return base_config

# Example usage function
def train_with_smart_stopping(env_name: str = "ALE/Pong-v5"):
    """Example of training with smart early stopping"""
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from stable_baselines3.common.env_util import make_vec_env
    
    # Create environment
    env = make_vec_env(env_name, n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    
    # Create evaluation environment
    eval_env = make_vec_env(env_name, n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    # Create training manager
    training_manager = AdaptiveTrainingManager(
        min_timesteps=50_000,
        max_timesteps=300_000,
        target_performance=-12.0 if "Pong" in env_name else 100.0,
        patience_episodes=50,
        improvement_threshold=2.0
    )
    
    # Create optimized model
    config = create_optimized_dqn_config(env_name)
    model = DQN("CnnPolicy", env, verbose=1, **config)
    
    # Create smart callback
    callback = SmartEvalCallback(eval_env, training_manager, eval_freq=10_000)
    
    print(f"🚀 Starting smart training for {env_name}")
    print(f"Max timesteps: {training_manager.max_timesteps}")
    print(f"Target performance: {training_manager.target_performance}")
    
    # Train with smart stopping
    start_time = time.time()
    model.learn(total_timesteps=training_manager.max_timesteps, callback=callback)
    total_time = time.time() - start_time
    
    # Generate report
    final_performance = training_manager.performance_history[-1] if training_manager.performance_history else 0
    efficiency = training_manager.get_efficiency_score()
    
    print(f"\n📊 TRAINING COMPLETE")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Final performance: {final_performance:.2f}")
    print(f"Efficiency score: {efficiency:.4f}")
    print(f"Timesteps used: {training_manager.timestep_history[-1] if training_manager.timestep_history else 0}")
    
    # Plot results
    training_manager.plot_training_progress()
    
    return model, training_manager
