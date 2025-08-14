#!/usr/bin/env python3
"""
Example usage of the mask visualization functions.
"""

import torch
from mask_visualization import (
    visualize_multi_head_masks,
    visualize_attention_mask_heatmap,
    quick_demo
)
from sub_models.attention_blocks import get_per_head_fixed_mask_causal

def example_1_basic_usage():
    """Example 1: Basic mask visualization."""
    print("Example 1: Basic Multi-Head Mask Visualization")
    print("-" * 50)
    
    # Set parameters
    batch_length = 8
    mask_percents = [0.0, 0.25, 0.5, 0.75]  # Different masking for each head
    device = torch.device('cpu')  # Use CPU for simplicity
    
    # Create and visualize masks
    fig = visualize_multi_head_masks(
        batch_length=batch_length,
        mask_percents=mask_percents,
        flag=False,  # Sequential masking (not random)
        soft=True,   # Soft penalties (not hard -inf)
        soft_penalty=-1.0,
        device=device,
        figsize=(16, 8)
    )
    
    # Save the plot
    fig.savefig('example_1_basic.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: example_1_basic.png")

def example_2_single_head_detailed():
    """Example 2: Detailed view of a single head."""
    print("\nExample 2: Detailed Single Head Analysis")
    print("-" * 50)
    
    batch_length = 12
    mask_percent = 0.4  # 40% masking
    device = torch.device('cpu')
    
    # Generate mask for single head
    mask = get_per_head_fixed_mask_causal(
        batch_length=batch_length,
        mask_percents=[mask_percent],
        flag=False,  # Sequential
        soft=True,   # Soft masking
        device=device,
        soft_penalty=-1.5
    )
    
    # Visualize single head
    fig = visualize_attention_mask_heatmap(
        mask=mask,
        title=f"Single Head: {mask_percent*100:.0f}% Sequential Soft Masking",
        head_idx=0,
        figsize=(10, 8)
    )
    
    fig.savefig('example_2_single_head.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: example_2_single_head.png")
    
    # Print some statistics
    mask_data = mask[0, 0].cpu().numpy()
    total_positions = batch_length * (batch_length + 1) // 2  # Lower triangle
    masked_positions = torch.sum(mask[0, 0] == float('-inf')).item()
    print(f"  Total causal positions: {total_positions}")
    print(f"  Masked positions: {masked_positions}")
    print(f"  Actual mask percentage: {masked_positions/total_positions*100:.1f}%")

def example_3_config_comparison():
    """Example 3: Compare different head configurations."""
    print("\nExample 3: Comparing Head Configurations")
    print("-" * 50)
    
    batch_length = 10
    device = torch.device('cpu')
    
    # Configuration 1: Uniform masking
    uniform_percents = [0.3, 0.3, 0.3, 0.3]
    fig1 = visualize_multi_head_masks(
        batch_length, uniform_percents, flag=False, soft=True,
        device=device, figsize=(16, 6)
    )
    fig1.suptitle("Uniform Masking: All heads 30%", fontsize=16)
    fig1.savefig('example_3a_uniform.png', dpi=200, bbox_inches='tight')
    
    # Configuration 2: Diversified masking
    diversified_percents = [0.0, 0.2, 0.5, 0.8]
    fig2 = visualize_multi_head_masks(
        batch_length, diversified_percents, flag=False, soft=True,
        device=device, figsize=(16, 6)
    )
    fig2.suptitle("Diversified Masking: 0%, 20%, 50%, 80%", fontsize=16)
    fig2.savefig('example_3b_diversified.png', dpi=200, bbox_inches='tight')
    
    print("✓ Saved: example_3a_uniform.png")
    print("✓ Saved: example_3b_diversified.png")

def example_4_quick_demo():
    """Example 4: Use the built-in quick demo."""
    print("\nExample 4: Quick Demo")
    print("-" * 50)
    
    # Run the built-in demo
    success = quick_demo(batch_length=8, save_plots=True)
    
    if success:
        print("✓ Quick demo completed")
    else:
        print("✗ Quick demo failed")

if __name__ == "__main__":
    print("Mask Visualization Examples")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_single_head_detailed()
        example_3_config_comparison()
        example_4_quick_demo()
        
        print("\n🎉 All examples completed successfully!")
        print("\nGenerated files:")
        print("  - example_1_basic.png")
        print("  - example_2_single_head.png") 
        print("  - example_3a_uniform.png")
        print("  - example_3b_diversified.png")
        print("  - demo_multi_head.png")
        print("  - demo_strategies.png")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()
