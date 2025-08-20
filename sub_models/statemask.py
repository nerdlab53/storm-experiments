import torch
import torch.nn as nn
import torch.nn.functional as F

class StateMaskGate(nn.Module):
    """Time-step action blinding gate (paper-aligned StateMask integration).

    Maps the agent's current latent (feature embedding) to a gate probability
    g_t in (0, 1). At rollout time, sample Bernoulli(g_t) to decide whether to
    pass the policy action through (g_t=1) or blind to a random action (g_t=0).

    This module is trained separately (e.g., periodically) to preserve the
    agent's action distribution (fidelity) while encouraging more blinding
    (sparsity), decoupled from the main RL objective.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, last_latent: torch.Tensor) -> torch.Tensor:
        """Returns gate probability in (0, 1) for each item in the batch.

        last_latent: [B, F] embedding of the current time step fed to the actor.
        """
        return torch.sigmoid(self.net(last_latent))  # [B, 1]

    @staticmethod
    def kl_fidelity_with_uniform(original_logits: torch.Tensor, gate_prob: torch.Tensor) -> torch.Tensor:
        """KL fidelity between original policy and a masked mixture with uniform.

        Approximates masked action distribution as a convex mixture of the
        original policy and a uniform distribution, weighted by gate_prob.
        """
        original_log_probs = original_logits.log_softmax(dim=-1)
        original_probs = original_log_probs.exp()
        num_actions = original_logits.shape[-1]
        uniform_probs = torch.full_like(original_probs, 1.0 / num_actions)
        mixed_probs = gate_prob * original_probs + (1.0 - gate_prob) * uniform_probs
        # KL(P || Q) = sum P log(P/Q)
        kl = (original_probs * (original_log_probs - mixed_probs.clamp_min(1e-8).log())).sum(dim=-1)
        return kl.mean()

    @staticmethod
    def sparsity_term(gate_prob: torch.Tensor) -> torch.Tensor:
        """Encourage more blinding: penalize pass-through probability.

        Lower gate_prob -> more blinding -> smaller penalty desired when masked.
        We penalize mean(gate_prob) to push towards masking non-critical steps.
        """
        return gate_prob.mean()
