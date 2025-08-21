import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple
from sub_models.statemask import StateMaskGate


class SimpleStateMaskTrainer:
    """Simple, modular StateMask trainer following the paper's true objective.
    
    Objective: minimize E[Return(π_surrogate)] where π_surrogate = StateMask ∘ Agent
    
    The surrogate policy works as:
    1. Agent produces action preferences (logits)
    2. StateMask decides whether to pass through (gate=1) or blind with random action (gate=0)
    3. We minimize the performance degradation of this surrogate policy
    """
    
    def __init__(
        self,
        statemask: StateMaskGate,
        agent_value_function,  # Agent's critic for estimating returns
        lr: float = 1e-4,
        target_sparsity: float = 0.3,
        sparsity_weight: float = 0.1
    ):
        self.statemask = statemask
        self.agent_value_function = agent_value_function
        self.optimizer = optim.Adam(statemask.parameters(), lr=lr)
        
        # Objectives
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
        
        # Simple experience collection
        self.experience_states = []
        self.experience_values = []
        self.max_buffer_size = 5000
    
    def collect_experience(self, states: torch.Tensor):
        """Collect state experiences and their values for StateMask training.
        
        Args:
            states: [B, feat_dim] - Agent state representations
        """
        with torch.no_grad():
            # Get current state values from agent's critic
            values = self.agent_value_function(states)
            
            # Store experiences
            self.experience_states.append(states.detach().cpu())
            self.experience_values.append(values.detach().cpu())
            
            # Maintain buffer size
            if len(self.experience_states) > self.max_buffer_size // states.shape[0]:
                self.experience_states.pop(0)
                self.experience_values.pop(0)
    
    def estimate_surrogate_value(self, states: torch.Tensor, original_values: torch.Tensor) -> torch.Tensor:
        """Estimate the value of states under the surrogate policy.
        
        Args:
            states: [B, feat_dim] - State representations
            original_values: [B] - Values under original agent policy
            
        Returns:
            estimated_surrogate_values: [B] - Estimated values under StateMask blinding
        """
        # Get gate probabilities
        gate_probs = self.statemask(states).squeeze(-1)  # [B]
        
        # Estimate value under blinding
        # When gate_prob = 1.0: keep original value
        # When gate_prob = 0.0: assume random policy (much lower value)
        
        # Simple heuristic: random policy gets ~10% of original value
        # This is a simplification - in reality we'd need more sophisticated estimation
        random_policy_value_ratio = 0.1
        blinded_values = original_values * random_policy_value_ratio
        
        # Linear interpolation based on gate probability
        surrogate_values = gate_probs * original_values + (1 - gate_probs) * blinded_values
        
        return surrogate_values
    
    def compute_return_based_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the paper's true objective: minimize expected return degradation.
        
        Returns:
            loss: Total loss to minimize
            metrics: Dictionary of training metrics
        """
        if len(self.experience_states) < 3:
            return torch.tensor(0.0, requires_grad=True), {}
        
        # Sample batch from experience
        batch_size = min(256, len(self.experience_states) * self.experience_states[0].shape[0])
        
        # Concatenate all experiences
        all_states = torch.cat(self.experience_states, dim=0).cuda()
        all_values = torch.cat(self.experience_values, dim=0).cuda()
        
        # Random sampling
        indices = torch.randperm(all_states.shape[0])[:batch_size]
        states = all_states[indices]
        original_values = all_values[indices]
        
        # Estimate surrogate policy values
        surrogate_values = self.estimate_surrogate_value(states, original_values)
        
        # Primary objective: minimize value degradation (maximize surrogate values)
        # We want high values, so we minimize negative values
        value_loss = -surrogate_values.mean()
        
        # Secondary objective: achieve target sparsity
        gate_probs = self.statemask(states).squeeze(-1)
        current_sparsity = 1.0 - gate_probs.mean()
        sparsity_loss = torch.abs(current_sparsity - self.target_sparsity)
        
        # Total loss
        total_loss = value_loss + self.sparsity_weight * sparsity_loss
        
        # Metrics
        metrics = {
            'statemask/value_loss': value_loss.item(),
            'statemask/sparsity_loss': sparsity_loss.item(),
            'statemask/total_loss': total_loss.item(),
            'statemask/gate_prob_mean': gate_probs.mean().item(),
            'statemask/current_sparsity': current_sparsity.item(),
            'statemask/target_sparsity': self.target_sparsity,
            'statemask/original_value_mean': original_values.mean().item(),
            'statemask/surrogate_value_mean': surrogate_values.mean().item(),
            'statemask/value_degradation': (original_values.mean() - surrogate_values.mean()).item()
        }
        
        return total_loss, metrics
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step of StateMask.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.experience_states) < 3:
            return {}
        
        self.statemask.train()
        
        # Compute loss based on return objective
        loss, metrics = self.compute_return_based_loss()
        
        if loss.item() == 0.0:
            return {}
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.statemask.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return metrics
    
    def evaluate(self, states: torch.Tensor) -> Dict[str, float]:
        """Evaluate StateMask on a batch of states.
        
        Args:
            states: [B, feat_dim] - States to evaluate on
            
        Returns:
            Evaluation metrics
        """
        self.statemask.eval()
        
        with torch.no_grad():
            gate_probs = self.statemask(states).squeeze(-1)
            original_values = self.agent_value_function(states)
            surrogate_values = self.estimate_surrogate_value(states, original_values)
            
            # Simulate actual blinding decisions
            binary_gates = torch.bernoulli(gate_probs)
            actual_sparsity = 1.0 - binary_gates.mean()
            
        return {
            'eval_statemask/gate_prob_mean': gate_probs.mean().item(),
            'eval_statemask/gate_prob_std': gate_probs.std().item(),
            'eval_statemask/actual_sparsity': actual_sparsity.item(),
            'eval_statemask/value_preservation': (surrogate_values.mean() / original_values.mean()).item(),
            'eval_statemask/value_degradation': (original_values.mean() - surrogate_values.mean()).item()
        }
    
    def set_target_sparsity(self, target: float):
        """Dynamically adjust target sparsity."""
        self.target_sparsity = max(0.0, min(1.0, target))


def create_simple_statemask_trainer(
    feat_dim: int, 
    agent_value_function,
    config: Dict
) -> Tuple[StateMaskGate, SimpleStateMaskTrainer]:
    """Factory function to create simple StateMask trainer.
    
    Args:
        feat_dim: Dimension of state features
        agent_value_function: Agent's value function for return estimation
        config: Configuration dictionary
        
    Returns:
        Tuple of (StateMaskGate, SimpleStateMaskTrainer)
    """
    # Create StateMask
    statemask = StateMaskGate(
        feat_dim=feat_dim,
        hidden_dim=config.get('hidden_dim', 128)
    )
    
    # Create simple trainer
    trainer = SimpleStateMaskTrainer(
        statemask=statemask,
        agent_value_function=agent_value_function,
        lr=config.get('lr', 1e-4),
        target_sparsity=config.get('target_sparsity', 0.3),
        sparsity_weight=config.get('sparsity_weight', 0.1)
    )
    
    return statemask, trainer
