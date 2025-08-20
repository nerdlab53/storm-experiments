import torch
import torch.nn.functional as F

def kl_fidelity_with_uniform(original_logits, gate_prob):
    """Standalone version of the KL fidelity function"""
    original_log_probs = original_logits.log_softmax(dim=-1)
    original_probs = original_log_probs.exp()
    num_actions = original_logits.shape[-1]
    uniform_probs = torch.full_like(original_probs, 1.0 / num_actions)
    mixed_probs = gate_prob * original_probs + (1.0 - gate_prob) * uniform_probs
    kl = (original_probs * (original_log_probs - mixed_probs.clamp_min(1e-8).log())).sum(dim=-1)
    return kl.mean()

def visualize_statemask_behavior():
    """
    Demonstrate the difference between:
    1. Actual masking behavior (Bernoulli sampling)
    2. KL fidelity approximation (smooth mixture)
    """
    
    print("=" * 60)
    print("STATEMASK BEHAVIOR VISUALIZATION")
    print("=" * 60)
    
    # Create small example tensors
    batch_size = 3
    num_actions = 4
    
    # Example action logits from a policy (e.g., [left, right, up, down])
    original_logits = torch.tensor([
        [2.0, 0.5, -1.0, 0.1],  # Strong preference for action 0 (left)
        [0.2, 0.2, 0.2, 3.0],   # Strong preference for action 3 (down) 
        [1.0, 1.0, 1.0, 1.0]    # Uniform preference
    ], dtype=torch.float32)
    
    # Gate probabilities (probability of using original policy vs masking)
    gate_probs = torch.tensor([[0.9], [0.3], [0.5]], dtype=torch.float32)  # [B, 1]
    
    print(f"Original logits shape: {original_logits.shape}")
    print(f"Gate probabilities shape: {gate_probs.shape}")
    print()
    
    # Convert logits to probabilities
    original_probs = torch.softmax(original_logits, dim=-1)
    
    print("ORIGINAL POLICY PROBABILITIES:")
    for i in range(batch_size):
        probs_str = ", ".join([f"{p:.3f}" for p in original_probs[i].tolist()])
        print(f"  Sample {i}: [{probs_str}] (gate_prob: {gate_probs[i].item():.1f})")
    print()
    
    # === PART 1: ACTUAL MASKING BEHAVIOR (what happens during rollout) ===
    print("1. ACTUAL MASKING BEHAVIOR (during rollout):")
    print("-" * 40)
    
    # Sample from Bernoulli distribution
    torch.manual_seed(42)  # For reproducibility
    mask_decisions = torch.bernoulli(gate_probs)  # 1 = use original, 0 = use uniform
    
    uniform_probs = torch.full_like(original_probs, 1.0 / num_actions)
    
    # Apply actual masking
    actual_masked_probs = torch.zeros_like(original_probs)
    for i in range(batch_size):
        if mask_decisions[i] == 1:
            actual_masked_probs[i] = original_probs[i]  # Use original policy
        else:
            actual_masked_probs[i] = uniform_probs[i]   # Use uniform policy
    
    decisions_str = ", ".join([f"{int(d)}" for d in mask_decisions.squeeze().tolist()])
    print(f"Mask decisions (1=original, 0=uniform): [{decisions_str}]")
    print("Actual masked probabilities:")
    for i in range(batch_size):
        decision = "ORIGINAL" if mask_decisions[i] == 1 else "UNIFORM"
        probs_str = ", ".join([f"{p:.3f}" for p in actual_masked_probs[i].tolist()])
        print(f"  Sample {i}: [{probs_str}] ({decision})")
    print()
    
    # === PART 2: KL FIDELITY APPROXIMATION (what the loss function computes) ===
    print("2. KL FIDELITY APPROXIMATION (for training):")
    print("-" * 40)
    
    # This is what the kl_fidelity_with_uniform function computes
    kl_loss = kl_fidelity_with_uniform(original_logits, gate_probs)
    
    # Let's manually compute the mixed probabilities to show the approximation
    mixed_probs = gate_probs * original_probs + (1.0 - gate_probs) * uniform_probs
    
    print("Mixed probabilities (smooth approximation):")
    for i in range(batch_size):
        probs_str = ", ".join([f"{p:.3f}" for p in mixed_probs[i].tolist()])
        print(f"  Sample {i}: [{probs_str}] (gate_prob: {gate_probs[i].item():.1f})")
    
    print(f"\nKL divergence (fidelity loss): {kl_loss.item():.4f}")
    print()
    
    # === PART 3: UNDERSTANDING THE DIFFERENCE ===
    print("3. KEY DIFFERENCES:")
    print("-" * 40)
    print("ACTUAL MASKING:")
    print("  - Hard decisions: Either 100% original OR 100% uniform")
    print("  - Non-differentiable (can't backprop through Bernoulli sampling)")
    print("  - Used during rollout/evaluation")
    print()
    print("KL FIDELITY APPROXIMATION:")
    print("  - Soft mixture: Weighted combination of original and uniform")
    print("  - Differentiable (can backprop to train the gate network)")
    print("  - Used during training to update gate probabilities")
    print()
    
    # === PART 4: SHOW HOW GATE PROBABILITIES AFFECT KL LOSS ===
    print("4. HOW GATE PROBABILITY AFFECTS KL LOSS:")
    print("-" * 40)
    
    test_gate_probs = torch.tensor([[1.0], [0.8], [0.5], [0.2], [0.0]])
    test_logits = original_logits[0:1].repeat(5, 1)  # Use first sample's logits
    
    for i, gate_p in enumerate([1.0, 0.8, 0.5, 0.2, 0.0]):
        kl = kl_fidelity_with_uniform(test_logits[i:i+1], test_gate_probs[i:i+1])
        print(f"  Gate prob {gate_p:.1f} -> KL loss: {kl.item():.4f}")
    
    print()
    print("INTERPRETATION:")
    print("  - High gate_prob (≈1.0) -> Low KL loss (high fidelity)")
    print("  - Low gate_prob (≈0.0) -> High KL loss (low fidelity)")
    print("  - Training balances fidelity vs sparsity (encouraging some masking)")

def simple_tensor_example():
    """Simple step-by-step example with manual calculations"""
    print("\n" + "=" * 60)
    print("SIMPLE STEP-BY-STEP EXAMPLE")
    print("=" * 60)
    
    # Simple case: 3 actions, 1 sample
    logits = torch.tensor([[2.0, 0.0, -1.0]])  # Strong preference for action 0
    gate_prob = torch.tensor([[0.7]])           # 70% chance to use original policy
    
    print("INPUT:")
    print(f"  Action logits: {logits.tolist()}")
    print(f"  Gate probability: {gate_prob.item():.1f}")
    print()
    
    # Step 1: Convert to probabilities
    original_probs = F.softmax(logits, dim=-1)
    probs_list = [f"{p:.3f}" for p in original_probs[0].tolist()]
    print("STEP 1: Convert logits to probabilities")
    print(f"  Original policy: [{', '.join(probs_list)}]")
    
    # Step 2: Create uniform distribution  
    num_actions = logits.shape[-1]
    uniform_probs = torch.full_like(original_probs, 1.0 / num_actions)
    uniform_list = [f"{p:.3f}" for p in uniform_probs[0].tolist()]
    print(f"  Uniform policy:  [{', '.join(uniform_list)}]")
    
    # Step 3: Create mixture
    mixed_probs = gate_prob * original_probs + (1.0 - gate_prob) * uniform_probs
    mixed_list = [f"{p:.3f}" for p in mixed_probs[0].tolist()]
    print(f"  Mixed policy:    [{', '.join(mixed_list)}]")
    print()
    
    # Step 4: Compute KL divergence
    print("STEP 2: Compute KL divergence")
    original_log_probs = F.log_softmax(logits, dim=-1)
    kl = (original_probs * (original_log_probs - mixed_probs.clamp_min(1e-8).log())).sum()
    print(f"  KL divergence: {kl.item():.4f}")
    
    print()
    print("WHAT THIS MEANS:")
    gate_pct = gate_prob.item() * 100
    uniform_pct = (1 - gate_prob.item()) * 100
    print(f"  - With gate_prob={gate_prob.item():.1f}, the mixed policy is {gate_pct:.0f}% original + {uniform_pct:.0f}% uniform")
    print(f"  - KL={kl.item():.4f} measures how different this mixture is from the original")
    print(f"  - Lower gate_prob → more uniform mixing → higher KL divergence")

if __name__ == "__main__":
    visualize_statemask_behavior()
    simple_tensor_example()
