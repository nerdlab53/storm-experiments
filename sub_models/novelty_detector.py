import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, kl_divergence
from collections import deque
import numpy as np
from typing import Dict, List, Optional, Tuple


class WorldModelNoveltyWrapper(nn.Module):
    """Wrapper adding novelty detection to WorldModel with integrated detection logic"""
    
    def __init__(self, world_model, 
                 history_length: int = 100,
                 detection_threshold_percentile: float = 95.0,
                 min_samples_for_detection: int = 50,
                 enable_adaptive_threshold: bool = True,
                 eig_threshold: float = 0.0,
                 use_eig_primary: bool = True,
                 warmup_steps: int = 1000):  # New param for warmup
        super().__init__()
        self.world_model = world_model
        
        self.history_length = history_length
        self.detection_threshold_percentile = detection_threshold_percentile
        self.min_samples_for_detection = min_samples_for_detection
        self.enable_adaptive_threshold = enable_adaptive_threshold
        self.eig_threshold = eig_threshold
        self.use_eig_primary = use_eig_primary
        self.warmup_steps = warmup_steps
        
        # History buffers
        self.kl_diff_history = deque(maxlen=history_length)
        self.expected_info_gain_history = deque(maxlen=history_length)
        
        # Stats
        self.total_steps = 0
        self.novelty_detections = 0
        self.detection_events = []
        self.current_threshold = 0.0
        self.initial_context = None
        
    def reset(self):
        """Reset detector state"""
        self.kl_diff_history.clear()
        self.expected_info_gain_history.clear()
        self.current_threshold = 0.0
        self.initial_context = None
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_steps': self.total_steps,
            'total_detections': self.novelty_detections,
            'detection_rate': self.novelty_detections / max(1, self.total_steps),
            'current_threshold': self.current_threshold,
            'recent_detections': self.detection_events[-5:]
        }
    
    def save_detection_log(self, filepath: str):
        """Save detection log"""
        import json
        with open(filepath, 'w') as f:
            json.dump({'statistics': self.get_statistics(), 'events': self.detection_events}, f)
    
    def forward(self, *args, **kwargs):
        return self.world_model(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.world_model, name)
    
    def detect_novelty_step(self, obs: torch.Tensor, action: torch.Tensor,
                          latent_context: torch.Tensor, current_step: int) -> Tuple[bool, Dict]:
        """Single method for complete novelty detection"""
        with torch.no_grad():
            # Set initial context on first call
            if self.initial_context is None:
                self.initial_context = latent_context.detach().clone()
                return False, {'message': 'Initial context set'}
            
            self.total_steps += 1
            
            if self.total_steps < self.warmup_steps:
                return False, {'message': f'In warmup (step {self.total_steps}/{self.warmup_steps}) - skipping detection'}
            
            # Get features and distributions
            obs_embedding = self.world_model.encoder(obs)  # B L (C H W)
            posterior_logits = self.world_model.dist_head.forward_post(obs_embedding).flatten(0, -2)  # p(z|h_t, x_t) approx
            prior_logits = self.world_model.dist_head.forward_prior(latent_context).flatten(0, -2)  # p(z|h_t)
            baseline_logits = self.world_model.dist_head.forward_prior(self.initial_context).flatten(0, -2)  # p(z|h_0)
            conditioned_post_logits = self.world_model.dist_head.forward_conditioned_post(obs_embedding, self.initial_context).flatten(0, -2)  # p(z|h_0, x_t)
            
            # Compute KL terms for bound
            posterior_dist = OneHotCategorical(logits=posterior_logits)
            prior_dist = OneHotCategorical(logits=prior_logits)
            baseline_dist = OneHotCategorical(logits=baseline_logits)
            cond_post_dist = OneHotCategorical(logits=conditioned_post_logits)
            
            kl_left = kl_divergence(posterior_dist, prior_dist).mean()  # KL[p(z|h_t,x_t) || p(z|h_t)]
            kl_to_baseline = kl_divergence(posterior_dist, baseline_dist).mean()  # KL[.. || p(z|h_0)]
            kl_to_cond_post = kl_divergence(posterior_dist, cond_post_dist).mean()  # KL[.. || p(z|h_0,x_t)]
            kl_bound_diff = kl_to_baseline - kl_to_cond_post  # RHS of bound
            bound_violated = kl_left > kl_bound_diff  # Check if KL_left > RHS (violation)
            
            # Compute EIG for additional signal
            posterior_probs = F.softmax(posterior_logits, dim=-1)
            prior_probs = F.softmax(prior_logits, dim=-1)
            baseline_probs = F.softmax(baseline_logits, dim=-1)
            epsilon = 1e-8
            log_ratio = torch.log((prior_probs + epsilon) / (baseline_probs + epsilon))
            eig = torch.sum(posterior_probs * log_ratio, dim=-1).mean()
            
            # For adaptive threshold, use bound_diff instead of previous kl_difference
            bound_diff = kl_bound_diff.item()
            self.kl_diff_history.append(bound_diff)
            self.expected_info_gain_history.append(eig.item())
            
            if self.enable_adaptive_threshold and len(self.kl_diff_history) >= self.min_samples_for_detection:
                self.current_threshold = np.percentile(
                    list(self.kl_diff_history), 100 - self.detection_threshold_percentile
                )
            
            # Detection logic: primary is bound violation or negative EIG
            is_novelty = False
            if len(self.kl_diff_history) >= self.min_samples_for_detection:
                eig_triggered = eig.item() < self.eig_threshold
                kl_triggered = bound_violated or (bound_diff < self.current_threshold)
                
                is_novelty = eig_triggered or kl_triggered if self.use_eig_primary else kl_triggered or eig_triggered
                
                if is_novelty:
                    self.novelty_detections += 1
                    detection_method = []
                    if eig_triggered:
                        detection_method.append('negative_eig')
                    if bound_violated:
                        detection_method.append('kl_bound_violation')
                        
                    self.detection_events.append({
                        'step': current_step,
                        'kl_left': kl_left.item(),
                        'kl_bound_diff': bound_diff,
                        'eig': eig.item(),
                        'detection_method': detection_method
                    })
            
            detection_info = {
                'kl_left': kl_left.item(),
                'kl_bound_diff': bound_diff,
                'expected_info_gain': eig.item(),
                'threshold': self.current_threshold,
                'eig_triggered': eig_triggered,
                'kl_bound_triggered': kl_triggered,
                'bound_violated': bound_violated,
                'total_detections': self.novelty_detections,
                'detection_rate': self.novelty_detections / max(1, self.total_steps)
            }
            
            return is_novelty, detection_info
    
    def reset_novelty_detection(self):
        """Reset novelty detection"""
        self.reset() 