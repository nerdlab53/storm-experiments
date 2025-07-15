import numpy as np
import torch
import cv2
from typing import Dict, Any, Callable, Optional
import gymnasium


class NoveltyInjector:
    """
    Novelty Injection System for Testing Novelty Detection
    
    Based on the novelties described in the paper:
    - Visual novelties (color changes, invisibility, noise)
    - Dynamic novelties (physics changes, behavior changes)
    - Structural novelties (environment layout changes)
    """
    
    def __init__(self):
        self.active_novelties = {}
        self.novelty_start_step = None
        self.novelty_config = None
        
    def inject_visual_noise(self, obs: np.ndarray, noise_std: float = 30.0) -> np.ndarray:
        """Inject Gaussian noise into observations"""
        noise = np.random.normal(0, noise_std, obs.shape).astype(obs.dtype)
        noisy_obs = np.clip(obs.astype(np.float32) + noise, 0, 255).astype(obs.dtype)
        return noisy_obs
    
    def inject_color_shift(self, obs: np.ndarray, shift_type: str = "red") -> np.ndarray:
        """Change color properties of observations"""
        modified_obs = obs.copy()
        
        if shift_type == "red":
            # Shift everything to red tones
            modified_obs[:, :, 1] = modified_obs[:, :, 1] * 0.3  # Reduce green
            modified_obs[:, :, 2] = modified_obs[:, :, 2] * 0.3  # Reduce blue
        elif shift_type == "blue":
            # Shift everything to blue tones  
            modified_obs[:, :, 0] = modified_obs[:, :, 0] * 0.3  # Reduce red
            modified_obs[:, :, 1] = modified_obs[:, :, 1] * 0.3  # Reduce green
        elif shift_type == "grayscale":
            # Convert to grayscale
            gray = np.dot(modified_obs[..., :3], [0.299, 0.587, 0.114])
            modified_obs = np.stack([gray, gray, gray], axis=-1).astype(obs.dtype)
        elif shift_type == "invert":
            # Invert colors
            modified_obs = 255 - modified_obs
            
        return modified_obs
    
    def inject_partial_invisibility(self, obs: np.ndarray, mask_ratio: float = 0.3) -> np.ndarray:
        """Make parts of the observation invisible (black)"""
        modified_obs = obs.copy()
        h, w = obs.shape[:2]
        
        # Create random mask
        mask = np.random.random((h, w)) < mask_ratio
        modified_obs[mask] = 0  # Make masked areas black
        
        return modified_obs
    
    def inject_blur(self, obs: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply blur to observations"""
        blurred = cv2.GaussianBlur(obs, (kernel_size, kernel_size), 0)
        return blurred
    
    def inject_rotation(self, obs: np.ndarray, angle: float = 15.0) -> np.ndarray:
        """Rotate observations"""
        h, w = obs.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(obs, M, (w, h))
        return rotated
    
    def configure_novelty(self, novelty_type: str, novelty_params: Dict[str, Any], start_step: int):
        """Configure a novelty to be injected at a specific step"""
        self.novelty_config = {
            'type': novelty_type,
            'params': novelty_params,
            'start_step': start_step
        }
        self.novelty_start_step = start_step
        
    def apply_novelty(self, obs: np.ndarray, current_step: int) -> np.ndarray:
        """Apply configured novelty if current step >= start step"""
        if (self.novelty_config is None or 
            self.novelty_start_step is None or 
            current_step < self.novelty_start_step):
            return obs
            
        novelty_type = self.novelty_config['type']
        params = self.novelty_config['params']
        
        if novelty_type == "visual_noise":
            return self.inject_visual_noise(obs, **params)
        elif novelty_type == "color_shift":
            return self.inject_color_shift(obs, **params)
        elif novelty_type == "partial_invisibility":
            return self.inject_partial_invisibility(obs, **params)
        elif novelty_type == "blur":
            return self.inject_blur(obs, **params)
        elif novelty_type == "rotation":
            return self.inject_rotation(obs, **params)
        else:
            print(f"Unknown novelty type: {novelty_type}")
            return obs
    
    def reset(self):
        """Reset novelty injection state"""
        self.active_novelties = {}
        self.novelty_start_step = None
        self.novelty_config = None


class NoveltyEnvironmentWrapper(gymnasium.Wrapper):
    """
    Environment wrapper that can inject novelties during episodes
    """
    
    def __init__(self, env, novelty_injector: Optional[NoveltyInjector] = None):
        super().__init__(env)
        self.novelty_injector = novelty_injector or NoveltyInjector()
        self.step_count = 0
        
    def reset(self, **kwargs):
        """Reset environment and novelty injection"""
        self.step_count = 0
        self.novelty_injector.reset()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step environment and apply novelties to observations"""
        obs, reward, done, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Apply novelty to observation
        modified_obs = self.novelty_injector.apply_novelty(obs, self.step_count)
        
        # Add novelty info to the info dict
        info['novelty_active'] = (
            self.novelty_injector.novelty_start_step is not None and 
            self.step_count >= self.novelty_injector.novelty_start_step
        )
        info['step_count'] = self.step_count
        
        return modified_obs, reward, done, truncated, info
    
    def configure_novelty(self, novelty_type: str, novelty_params: Dict[str, Any], start_step: int):
        """Configure novelty injection"""
        self.novelty_injector.configure_novelty(novelty_type, novelty_params, start_step)


# Predefined novelty configurations from the paper
PREDEFINED_NOVELTIES = {
    # Visual novelties
    "atari_invisible_cars": {
        "type": "partial_invisibility",
        "params": {"mask_ratio": 0.8},
        "start_step": 100
    },
    "atari_color_cars": {
        "type": "color_shift", 
        "params": {"shift_type": "red"},
        "start_step": 100
    },
    "dmc_noise_low": {
        "type": "visual_noise",
        "params": {"noise_std": 5.0},
        "start_step": 30
    },
    "dmc_noise_high": {
        "type": "visual_noise", 
        "params": {"noise_std": 30.0},
        "start_step": 30
    },
    "general_blur": {
        "type": "blur",
        "params": {"kernel_size": 15},
        "start_step": 50
    },
    "general_rotation": {
        "type": "rotation",
        "params": {"angle": 15.0},
        "start_step": 50
    },
    "general_grayscale": {
        "type": "color_shift",
        "params": {"shift_type": "grayscale"},
        "start_step": 50
    },
    "general_invert": {
        "type": "color_shift",
        "params": {"shift_type": "invert"},
        "start_step": 50
    }
}


def create_novelty_test_scenarios() -> Dict[str, Dict]:
    """Create a comprehensive set of novelty test scenarios"""
    scenarios = {}
    
    # Gradual noise introduction
    for noise_level in [5, 10, 20, 30]:
        scenarios[f"noise_std_{noise_level}"] = {
            "type": "visual_noise",
            "params": {"noise_std": float(noise_level)},
            "start_step": 100
        }
    
    # Different invisibility levels
    for mask_ratio in [0.1, 0.3, 0.5, 0.8]:
        scenarios[f"invisibility_{int(mask_ratio*100)}"] = {
            "type": "partial_invisibility", 
            "params": {"mask_ratio": mask_ratio},
            "start_step": 100
        }
    
    # Different blur levels
    for kernel_size in [5, 10, 15, 25]:
        scenarios[f"blur_kernel_{kernel_size}"] = {
            "type": "blur",
            "params": {"kernel_size": kernel_size},
            "start_step": 100
        }
    
    # Different rotation angles
    for angle in [5, 10, 15, 30, 45]:
        scenarios[f"rotation_{angle}"] = {
            "type": "rotation",
            "params": {"angle": float(angle)},
            "start_step": 100
        }
    
    # Color shifts
    for shift_type in ["red", "blue", "grayscale", "invert"]:
        scenarios[f"color_{shift_type}"] = {
            "type": "color_shift",
            "params": {"shift_type": shift_type},
            "start_step": 100
        }
    
    return scenarios


def test_novelty_detection_system(env_name: str,
                                world_model,
                                agent,
                                novelty_scenarios: Dict[str, Dict],
                                episodes_per_scenario: int = 5) -> Dict[str, Any]:
    """
    Comprehensive test of novelty detection system across multiple scenarios
    
    Args:
        env_name: Environment name
        world_model: Trained world model with novelty detection
        agent: Trained agent
        novelty_scenarios: Dictionary of novelty configurations
        episodes_per_scenario: Number of episodes to test per scenario
        
    Returns:
        Dictionary with test results
    """
    from train import build_single_env
    
    results = {}
    
    for scenario_name, scenario_config in novelty_scenarios.items():
        print(f"Testing scenario: {scenario_name}")
        scenario_results = []
        
        for episode in range(episodes_per_scenario):
            # Create environment with novelty injection
            base_env = build_single_env(env_name, 64, seed=42 + episode)
            env = NoveltyEnvironmentWrapper(base_env)
            
            # Configure novelty
            env.configure_novelty(
                scenario_config['type'],
                scenario_config['params'], 
                scenario_config['start_step']
            )
            
            # Reset world model novelty detection
            if hasattr(world_model, 'reset_novelty_detection'):
                world_model.reset_novelty_detection()
            
            # Run episode
            obs, info = env.reset()
            episode_detections = []
            total_steps = 0
            
            for step in range(1000):  # Max steps per episode
                # Convert obs to tensor format
                obs_tensor = torch.tensor(obs).unsqueeze(0).unsqueeze(0).float() / 255.0
                obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3)  # B, L, C, H, W
                
                # Get action from agent (simplified)
                with torch.no_grad():
                    # This is a simplified version - in practice you'd need the full context
                    action = env.action_space.sample()
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                total_steps += 1
                
                # Perform novelty detection (if world model supports it)
                if hasattr(world_model, 'detect_novelty_step'):
                    try:
                        # This would need proper implementation with actual latent context
                        is_novelty, detection_info = world_model.detect_novelty_step(
                            obs_tensor.cuda() if torch.cuda.is_available() else obs_tensor,
                            torch.tensor([action]),
                            torch.zeros(1, 1, 512).cuda() if torch.cuda.is_available() else torch.zeros(1, 1, 512),  # Dummy context
                            total_steps
                        )
                        
                        if is_novelty:
                            episode_detections.append({
                                'step': total_steps,
                                'novelty_active': info.get('novelty_active', False),
                                'detection_info': detection_info
                            })
                    except Exception as e:
                        print(f"Novelty detection error: {e}")
                
                if done or truncated:
                    break
            
            scenario_results.append({
                'episode': episode,
                'total_steps': total_steps,
                'detections': episode_detections,
                'novelty_start_step': scenario_config['start_step']
            })
        
        results[scenario_name] = scenario_results
    
    return results 