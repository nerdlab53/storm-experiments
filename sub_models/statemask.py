import torch
import torch.nn as nn
import torch.nn.functional as F


class StateMaskGate(nn.Module):
    """Binary gate that decides whether to pass-through or blind the agent's action.

    Forward returns a probability in [0, 1] for pass-through (shape: [B, 1]).
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def kl_fidelity_with_uniform(logits: torch.Tensor, gate_probs: torch.Tensor) -> torch.Tensor:
        """KL(p_agent || p_masked) where p_masked = g * p_agent + (1-g) * Uniform.

        Args:
            logits: [B, A] agent policy logits
            gate_probs: [B, 1] pass-through probability per sample

        Returns:
            Scalar KL divergence (batchmean)
        """
        probs_agent = F.softmax(logits, dim=-1)
        num_actions = probs_agent.shape[-1]
        uniform = torch.full_like(probs_agent, 1.0 / float(num_actions))
        # Broadcast gate over actions
        gate = gate_probs.clamp(0.0, 1.0)
        while gate.dim() < probs_agent.dim():
            gate = gate.unsqueeze(-1)
        probs_masked = gate * probs_agent + (1.0 - gate) * uniform
        # Add small epsilon for numerical stability
        eps = 1e-8
        kl = F.kl_div((probs_agent + eps).log(), probs_masked + eps, reduction='batchmean')
        return kl


