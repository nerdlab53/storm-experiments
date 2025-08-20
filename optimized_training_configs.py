"""
Optimized training configurations for efficient DQN learning
Addresses the trade-off between training time and performance
"""

# Strategy 1: Progressive Training Schedules
PROGRESSIVE_SCHEDULES = {
    "quick_test": {
        "description": "Fast initial assessment",
        "timesteps": 50_000,
        "eval_freq": 5_000,
        "target_performance": -18.0,  # Pong: stop early if achieving good performance
    },
    
    "medium_train": {
        "description": "Balanced training",
        "timesteps": 200_000,
        "eval_freq": 10_000,
        "target_performance": -15.0,
    },
    
    "full_train": {
        "description": "Full training if needed",
        "timesteps": 500_000,
        "eval_freq": 20_000,
        "target_performance": -10.0,
    }
}

# Strategy 2: Optimized Hyperparameters for Faster Learning
FAST_LEARNING_CONFIG = {
    "learning_rate": 2.5e-4,  # Higher LR for faster initial learning
    "buffer_size": 50_000,    # Larger buffer for better sample efficiency
    "learning_starts": 5_000, # Start learning earlier
    "batch_size": 64,         # Larger batches for more stable gradients
    "train_freq": 1,          # Train every step (more frequent updates)
    "gradient_steps": 2,      # Multiple gradient steps per update
    "target_update_interval": 1_000,  # More frequent target updates
    "exploration_fraction": 0.05,     # Shorter exploration phase
    "exploration_final_eps": 0.005,   # Lower final epsilon
}

# Strategy 3: Environment-Specific Optimizations
ENV_SPECIFIC_CONFIGS = {
    "ALE/Pong-v5": {
        "frame_stack": 4,     # 4 frames sufficient for Pong
        "frame_skip": 4,      # Skip frames to speed up training
        "expected_episodes": 2000,  # Typical episodes to convergence
        "early_stop_reward": -5.0,  # Stop if consistently achieving this
    },
    
    "ALE/Breakout-v5": {
        "frame_stack": 4,
        "frame_skip": 4,
        "expected_episodes": 5000,
        "early_stop_reward": 50.0,
    },
    
    "ALE/MsPacman-v5": {
        "frame_stack": 4,
        "frame_skip": 4,
        "expected_episodes": 3000,
        "early_stop_reward": 500.0,
    }
}

# Strategy 4: Multi-Stage Training
MULTI_STAGE_TRAINING = {
    "stage1_exploration": {
        "timesteps": 30_000,
        "exploration_fraction": 0.8,  # Heavy exploration initially
        "learning_rate": 1e-4,
        "target_update_interval": 5_000,
    },
    
    "stage2_intensive": {
        "timesteps": 100_000,
        "exploration_fraction": 0.1,  # Reduce exploration
        "learning_rate": 5e-4,        # Increase learning rate
        "target_update_interval": 2_000,
    },
    
    "stage3_refinement": {
        "timesteps": 70_000,
        "exploration_fraction": 0.05, # Minimal exploration
        "learning_rate": 1e-4,        # Lower LR for fine-tuning
        "target_update_interval": 1_000,
    }
}

# Strategy 5: Parallel Training with Different Configs
PARALLEL_EXPERIMENTS = [
    {
        "name": "aggressive",
        "config": {**FAST_LEARNING_CONFIG, "learning_rate": 5e-4},
        "timesteps": 150_000,
    },
    {
        "name": "conservative", 
        "config": {**FAST_LEARNING_CONFIG, "learning_rate": 1e-4},
        "timesteps": 150_000,
    },
    {
        "name": "balanced",
        "config": FAST_LEARNING_CONFIG,
        "timesteps": 150_000,
    }
]
