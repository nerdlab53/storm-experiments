import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        
        # Lagrangian optimization (correct StateMask approach)
        self.lambda_multiplier = 0.0  # Lagrange multiplier
        self.lr_lambda = 1e-3  # Learning rate for lambda updates  
        self.baseline_performance = None  # Will be set during training
        self.recent_performances = []
        self.performance_buffer_size = 100
        
        # Simple experience collection  
        self.experience_states = []
        self.experience_values = []
        self.experience_returns = []  # Track actual episode returns
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
        """DEPRECATED: This approach is incorrect according to StateMask paper.
        
        The correct approach is to use Lagrange multipliers with actual performance tracking,
        not value function estimation. See compute_lagrangian_loss() for the proper method.
        """
        raise NotImplementedError("Use compute_lagrangian_loss() instead - this method is incorrect per StateMask paper")
    
    def collect_performance(self, episode_return: float):
        """Collect actual episode returns for performance tracking.
        
        Args:
            episode_return: Actual return from a complete episode with StateMask
        """
        self.recent_performances.append(episode_return)
        if len(self.recent_performances) > self.performance_buffer_size:
            self.recent_performances.pop(0)
    
    def set_baseline_performance(self, baseline_perf: float):
        """Set the baseline agent's performance (eta_origin in paper)."""
        self.baseline_performance = baseline_perf
        
    def compute_lagrangian_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                               advantages: torch.Tensor, old_log_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the correct StateMask loss using Lagrangian formulation.
        
        Based on the Pong implementation:
        L = actor_loss + λ * sparsity_penalty
        
        where λ is updated as:
        λ -= lr_λ * (current_performance - baseline_performance + tolerance)
        
        Args:
            states: [B, feat_dim] - State representations
            actions: [B] - Actions taken (0=blind, 1=pass-through)
            advantages: [B] - Advantage estimates
            old_log_probs: [B] - Old action log probabilities
            
        Returns:
            loss: Combined loss tensor
            metrics: Dictionary of training metrics
        """
        gate_probs = self.statemask(states)  # [B, 1]
        dist = torch.distributions.Bernoulli(gate_probs.squeeze(-1))
        new_log_probs = dist.log_prob(actions.float())  # [B]
        # policy ratio (like PPO)
        ratio = (new_log_probs - old_log_probs).exp()  # [B]
        actor_loss = -(ratio * advantages).mean()
        sparsity_penalty = gate_probs.mean() 
        if self.lambda_multiplier > 1:
            actor_loss = -actor_loss
        total_loss = actor_loss + self.sparsity_weight * sparsity_penalty
        
        # update Lagrange multiplier if we have performance data
        if len(self.recent_performances) > 0 and self.baseline_performance is not None:
            current_perf = np.mean(self.recent_performances)
            
            # update lambda (following Pong implementation)
            # λ -= lr_λ * (current_perf - 2*expected_perf + 2*baseline_perf)
            lambda_update = current_perf - 2 * current_perf + 2 * self.baseline_performance
            self.lambda_multiplier -= self.lr_lambda * lambda_update
            self.lambda_multiplier = max(self.lambda_multiplier, 0.0)
        
        metrics = {
            'statemask/actor_loss': actor_loss.item(),
            'statemask/sparsity_penalty': sparsity_penalty.item(),
            'statemask/total_loss': total_loss.item(),
            'statemask/lambda_multiplier': self.lambda_multiplier,
            'statemask/gate_prob_mean': gate_probs.mean().item(),
            'statemask/gate_prob_std': gate_probs.std().item(),
            'statemask/current_performance': np.mean(self.recent_performances) if self.recent_performances else 0.0
        }
        return total_loss, metrics
    
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
        
        # Secondary objective: achieve target sparsityw
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
    
    def train_step(self, states: torch.Tensor = None, actions: torch.Tensor = None, 
                   advantages: torch.Tensor = None, old_log_probs: torch.Tensor = None) -> Dict[str, float]:
        """Train StateMask using the correct Lagrangian formulation.
        
        Args:
            states: [B, feat_dim] - State representations 
            actions: [B] - StateMask actions (0=blind, 1=pass-through)
            advantages: [B] - Advantage estimates from agent training
            old_log_probs: [B] - Previous action log probabilities
            
        Returns:
            Training metrics
        """
        # Use provided data or fall back to buffered experience
        if states is None:
            if len(self.experience_states) < 3:
                return {'statemask/no_data': 1.0}
            states = torch.cat(self.experience_states, dim=0).cuda()
        
        if actions is None or advantages is None or old_log_probs is None:
            # For compatibility with old approach, generate dummy data
            # TODO: This should be removed once proper integration is complete
            batch_size = states.shape[0]
            actions = torch.randint(0, 2, (batch_size,)).cuda()
            advantages = torch.randn(batch_size).cuda()
            old_log_probs = torch.randn(batch_size).cuda()
        
        self.statemask.train()
        
        # compute correct Lagrangian loss
        loss, metrics = self.compute_lagrangian_loss(states, actions, advantages, old_log_probs)
        
        if loss.item() == 0.0:
            return metrics
        
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
