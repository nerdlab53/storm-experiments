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
        clip_param: float = 0.2,
        gate_entropy_coef: float = 0.01,
        feat_dim: int = None,
        use_mask_critic: bool = True,
        mask_critic_hidden_dim: int = 128,
        value_coef: float = 0.5,
        lasso_weight: float = 1e-4
    ):
        self.statemask = statemask
        self.agent_value_function = agent_value_function
        self.value_coef = value_coef
        
        # Loss coefficients
        self.clip_param = clip_param
        self.gate_entropy_coef = gate_entropy_coef
        self.lasso_weight = lasso_weight
        
        # Device from statemask params
        try:
            self.device = next(statemask.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optional critic for masked policy value estimation
        self.use_mask_critic = use_mask_critic
        self.mask_value = None
        if self.use_mask_critic:
            assert feat_dim is not None, "feat_dim is required when use_mask_critic=True"
            self.mask_value = nn.Sequential(
                nn.Linear(feat_dim, mask_critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(mask_critic_hidden_dim, 1)
            ).to(self.device)
            params = list(statemask.parameters()) + list(self.mask_value.parameters())
        else:
            params = list(statemask.parameters())
        
        self.optimizer = optim.Adam(params, lr=lr)
    
    # No experience/performance bookkeeping in the simple version
        
    def ppo_update_sequential(self,
                              states: torch.Tensor,
                              actions: torch.Tensor,
                              old_log_probs: torch.Tensor,
                              advantages: torch.Tensor,
                              old_values: Optional[torch.Tensor] = None,
                              returns: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Sequential, minimal PPO-style update for the StateMask gate.

        This mirrors a simple step-by-step implementation:
        1) Forward -> gate_probs, dist
        2) Log probs -> ratio
        3) Clipped actor loss
        4) Optional critic loss
        5) Entropy bonus
        6) Pass-through penalty (lasso)
        7) Backprop + step
        """
        self.statemask.train()
        gate_probs = self.statemask(states)  # [B, 1]
        dist = torch.distributions.Bernoulli(gate_probs.squeeze(-1))
        new_log_probs = dist.log_prob(actions.float())  # [B]
        ratio = (new_log_probs - old_log_probs).exp()  # [B]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = torch.tensor(0.0, device=states.device)
        if self.use_mask_critic and self.mask_value is not None:
            values = self.mask_value(states).squeeze(-1)
            if returns is None:
                assert old_values is not None, "old_values or returns required for critic"
                returns = (advantages + old_values).detach()
            critic_loss = (returns - values).pow(2).mean()
        gate_entropy = dist.entropy().mean()
        pass_through_penalty = gate_probs.mean()
        total_loss = self.value_coef * critic_loss + actor_loss - self.gate_entropy_coef * gate_entropy + self.lasso_weight * pass_through_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.statemask.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics = {
            'statemask/actor_loss': actor_loss.item(),
            'statemask/critic_loss': float(critic_loss.item()) if isinstance(critic_loss, torch.Tensor) else float(critic_loss),
            'statemask/entropy': gate_entropy.item(),
            'statemask/num_masks': gate_probs.mean().item(),
            'statemask/total_loss': total_loss.item(),
        }
        return metrics

    # Removed alternative loss paths for simplicity
    
    def train_step(self, states: torch.Tensor = None, actions: torch.Tensor = None, 
                   advantages: torch.Tensor = None, old_log_probs: torch.Tensor = None,
                   old_values: torch.Tensor = None, returns: torch.Tensor = None) -> Dict[str, float]:
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
            if self.use_mask_critic:
                old_values = torch.randn(batch_size).cuda()
                returns = (advantages + old_values)
        
        self.statemask.train()
        
        # Perform a simple sequential PPO-style update
        metrics = self.ppo_update_sequential(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            old_values=old_values,
            returns=returns,
        )
        return metrics
    
    # No custom evaluation in the simple version
    
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
        clip_param=config.get('clip_param', 0.2),
        gate_entropy_coef=config.get('gate_entropy_coef', 0.01),
        feat_dim=feat_dim,
        use_mask_critic=config.get('use_mask_critic', True),
        mask_critic_hidden_dim=config.get('mask_critic_hidden_dim', 128),
        value_coef=config.get('value_coef', 0.5),
        lasso_weight=config.get('lasso_weight', 1e-4)
    )
    
    return statemask, trainer
