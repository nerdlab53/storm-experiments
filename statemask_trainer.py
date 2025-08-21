import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
import numpy as np
from sub_models.statemask import StateMaskGate


# this is a minimal implementation

class StateMaskTrainer:
    """Decoupled training for StateMask gate to balance fidelity and sparsity.
    
    This trainer optimizes the StateMask gate independently from the main RL
    objective, using collected experience to learn when to blind actions while
    preserving policy fidelity.
    """
    
    def __init__(
        self, 
        statemask: StateMaskGate,
        lr: float = 1e-4,
        fidelity_weight: float = 1.0,
        sparsity_weight: float = 0.1,
        target_sparsity: float = 0.3,
        train_frequency: int = 100,
        batch_size: int = 256,
        target_update_frequency: int = 1000,  # Slower target updates
        target_tau: float = 0.005  # Soft update coefficient for target network
    ):
        self.statemask = statemask
        self.optimizer = optim.Adam(statemask.parameters(), lr=lr)
        
        # Create target network for slower updates - extract parameters from main network
        feat_dim = statemask.net[0].in_features
        hidden_dim = statemask.net[0].out_features
        self.target_statemask = StateMaskGate(feat_dim, hidden_dim)
        self.target_statemask.load_state_dict(statemask.state_dict())
        self.target_statemask.eval()
        
        # Target network update parameters
        self.target_update_frequency = target_update_frequency
        self.target_tau = target_tau
        self.target_update_counter = 0
        
        # Loss weighting
        self.fidelity_weight = fidelity_weight
        self.sparsity_weight = sparsity_weight
        self.target_sparsity = target_sparsity
        
        # Training schedule
        self.train_frequency = train_frequency
        self.batch_size = batch_size
        self.step_count = 0
        
        # Experience buffer for decoupled training
        self.experience_buffer = {
            'states': [],
            'logits': [],
            'actions': []
        }
        self.buffer_size = 10000
        
    def collect_experience(self, states: torch.Tensor, logits: torch.Tensor, actions: torch.Tensor):
        """Collect experience from agent rollouts for later StateMask training.
        
        Args:
            states: [B, feat_dim] - agent state representations
            logits: [B, action_dim] - agent policy logits
            actions: [B] - sampled actions
        """
        batch_size = states.shape[0]
        
        # Add to buffer
        self.experience_buffer['states'].append(states.detach().cpu())
        self.experience_buffer['logits'].append(logits.detach().cpu())
        self.experience_buffer['actions'].append(actions.detach().cpu())
        
        # Maintain buffer size
        for key in self.experience_buffer:
            if len(self.experience_buffer[key]) > self.buffer_size // batch_size:
                self.experience_buffer[key].pop(0)
    
    def should_train(self) -> bool:
        """Check if it's time to train the StateMask."""
        self.step_count += 1
        return (self.step_count % self.train_frequency == 0 and 
                len(self.experience_buffer['states']) > 5)
    
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch from the experience buffer."""
        # Concatenate all buffer data
        all_states = torch.cat(self.experience_buffer['states'], dim=0)
        all_logits = torch.cat(self.experience_buffer['logits'], dim=0)
        all_actions = torch.cat(self.experience_buffer['actions'], dim=0)
        
        # Random sampling
        total_samples = all_states.shape[0]
        batch_size = min(self.batch_size, total_samples)
        indices = torch.randperm(total_samples)[:batch_size]
        
        return (
            all_states[indices].cuda(),
            all_logits[indices].cuda(), 
            all_actions[indices].cuda()
        )
    
    def train_step(self) -> dict:
        """Perform one training step of the StateMask gate.
        
        Returns:
            Dictionary with training metrics
        """
        if not self.should_train():
            return {}
            
        self.statemask.train()
        
        # Sample batch from experience buffer
        states, logits, actions = self.sample_batch()
        
        # Forward pass through StateMask
        gate_probs = self.statemask(states)  # [B, 1]
        
        # Compute fidelity loss (KL divergence with uniform mixture)
        fidelity_loss = StateMaskGate.kl_fidelity_with_uniform(logits, gate_probs)
        
        # Compute sparsity loss (encourage blinding)
        current_sparsity = 1.0 - gate_probs.mean()  # Higher when more blinding
        sparsity_loss = torch.abs(current_sparsity - self.target_sparsity)
        
        # Total loss
        total_loss = (self.fidelity_weight * fidelity_loss + 
                     self.sparsity_weight * sparsity_loss)
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.statemask.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network (slower updates)
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_frequency == 0:
            self._hard_update_target()
        else:
            self._soft_update_target()
        
        # Return metrics
        metrics = {
            'statemask/total_loss': total_loss.item(),
            'statemask/fidelity_loss': fidelity_loss.item(),
            'statemask/sparsity_loss': sparsity_loss.item(),
            'statemask/gate_prob_mean': gate_probs.mean().item(),
            'statemask/current_sparsity': current_sparsity.item(),
            'statemask/target_sparsity': self.target_sparsity,
            'statemask/target_update_counter': self.target_update_counter,
        }
        
        return metrics
    
    def evaluate_masking_quality(self, states: torch.Tensor, logits: torch.Tensor) -> dict:
        """Evaluate the quality of StateMask predictions on a batch.
        
        Args:
            states: [B, feat_dim] - state representations
            logits: [B, action_dim] - policy logits
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.statemask.eval()
        
        with torch.no_grad():
            gate_probs = self.statemask(states)
            fidelity_loss = StateMaskGate.kl_fidelity_with_uniform(logits, gate_probs)
            sparsity = 1.0 - gate_probs.mean()
            
            # Distribution of gate probabilities
            gate_std = gate_probs.std()
            gate_min = gate_probs.min()
            gate_max = gate_probs.max()
            
        return {
            'eval_statemask/fidelity_loss': fidelity_loss.item(),
            'eval_statemask/sparsity': sparsity.item(),
            'eval_statemask/gate_prob_mean': gate_probs.mean().item(),
            'eval_statemask/gate_prob_std': gate_std.item(),
            'eval_statemask/gate_prob_min': gate_min.item(),
            'eval_statemask/gate_prob_max': gate_max.item(),
        }
    
    def set_target_sparsity(self, target: float):
        """Dynamically adjust target sparsity during training."""
        self.target_sparsity = max(0.0, min(1.0, target))
    
    def _soft_update_target(self):
        """Soft update of target network parameters."""
        with torch.no_grad():
            for target_param, param in zip(self.target_statemask.parameters(), self.statemask.parameters()):
                target_param.data.copy_(
                    self.target_tau * param.data + (1.0 - self.target_tau) * target_param.data
                )
    
    def _hard_update_target(self):
        """Hard update of target network parameters."""
        self.target_statemask.load_state_dict(self.statemask.state_dict())
    
    def get_blinding_statistics(self, states: torch.Tensor) -> dict:
        """Get statistics about which states would be blinded."""
        self.statemask.eval()
        
        with torch.no_grad():
            gate_probs = self.statemask(states)
            binary_masks = torch.bernoulli(gate_probs)
            
        return {
            'blinding_rate': (1.0 - binary_masks.mean()).item(),
            'gate_variance': gate_probs.var().item(),
            'num_always_pass': (gate_probs > 0.9).sum().item(),
            'num_always_blind': (gate_probs < 0.1).sum().item(),
        }


def create_statemask_trainer(feat_dim: int, config: dict) -> Tuple[StateMaskGate, StateMaskTrainer]:
    """Factory function to create StateMask and its trainer.
    
    Args:
        feat_dim: Dimension of state features (32*32 + transformer_hidden_dim)
        config: Configuration dictionary with StateMask parameters
        
    Returns:
        Tuple of (StateMaskGate, StateMaskTrainer)
    """
    # Create StateMask gate
    statemask = StateMaskGate(
        feat_dim=feat_dim,
        hidden_dim=config.get('hidden_dim', 128)
    )
    
    # Create trainer
    trainer = StateMaskTrainer(
        statemask=statemask,
        lr=config.get('lr', 1e-4),
        fidelity_weight=config.get('fidelity_weight', 1.0),
        sparsity_weight=config.get('sparsity_weight', 0.1),
        target_sparsity=config.get('target_sparsity', 0.3),
        train_frequency=config.get('train_frequency', 100),
        batch_size=config.get('batch_size', 256),
        target_update_frequency=config.get('target_update_frequency', 1000),
        target_tau=config.get('target_tau', 0.005)
    )
    
    return statemask, trainer
