#!/usr/bin/env python3
"""
Simple test script for mask visualization functions.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mask_visualization import (
        visualize_multi_head_masks, 
        compare_masking_strategies,
        visualize_mask_percentage_effects,
        visualize_attention_mask_heatmap
    )
    from sub_models.attention_blocks import get_per_head_fixed_mask_causal
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic mask generation and visualization."""
    print("\n=== Testing Basic Functionality ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_length = 8
    mask_percents = [0.0, 0.25, 0.5, 0.75]
    
    try:
        # Test mask generation
        print("Testing mask generation...")
        mask = get_per_head_fixed_mask_causal(
            batch_length, mask_percents, 
            flag=False, soft=True, device=device, soft_penalty=-1.0
        )
        print(f"✓ Generated mask shape: {mask.shape}")
        
        # Test single head visualization
        print("Testing single head visualization...")
        fig1 = visualize_attention_mask_heatmap(
            mask, title="Test Single Head", head_idx=0, figsize=(8, 6)
        )
        fig1.savefig('test_single_head.png', dpi=150, bbox_inches='tight')
        print("✓ Single head visualization successful")
        
        # Test multi-head visualization
        print("Testing multi-head visualization...")
        fig2 = visualize_multi_head_masks(
            batch_length, mask_percents, 
            flag=False, soft=True, soft_penalty=-1.0, 
            device=device, figsize=(12, 8)
        )
        fig2.savefig('test_multi_head.png', dpi=150, bbox_inches='tight')
        print("✓ Multi-head visualization successful")
        
        # Test strategy comparison
        print("Testing strategy comparison...")
        fig3 = compare_masking_strategies(batch_length, device=device, figsize=(12, 8))
        fig3.savefig('test_strategies.png', dpi=150, bbox_inches='tight')
        print("✓ Strategy comparison successful")
        
        # Test percentage effects
        print("Testing percentage effects...")
        fig4 = visualize_mask_percentage_effects(batch_length, device=device, figsize=(12, 8))
        fig4.savefig('test_percentages.png', dpi=150, bbox_inches='tight')
        print("✓ Percentage effects visualization successful")
        
        print("\n✓ All tests passed! Generated images:")
        print("  - test_single_head.png")
        print("  - test_multi_head.png") 
        print("  - test_strategies.png")
        print("  - test_percentages.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mask_properties():
    """Test that masks have expected properties."""
    print("\n=== Testing Mask Properties ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_length = 6
    mask_percents = [0.0, 0.5, 1.0]
    
    mask = get_per_head_fixed_mask_causal(
        batch_length, mask_percents, 
        flag=False, soft=True, device=device, soft_penalty=-2.0
    )
    
    print(f"Mask shape: {mask.shape}")
    print(f"Expected shape: (1, {len(mask_percents)}, {batch_length}, {batch_length})")
    
    for h, percent in enumerate(mask_percents):
        head_mask = mask[0, h].cpu().numpy()
        
        # Check causal property
        upper_tri = torch.triu(torch.ones(batch_length, batch_length), diagonal=1).numpy()
        upper_tri_values = head_mask[upper_tri.astype(bool)]
        all_inf = np.all(upper_tri_values == float('-inf'))
        print(f"Head {h} ({percent*100:.0f}%): Causal property {'✓' if all_inf else '✗'}")
        
        # Check masking percentage (approximately)
        lower_tri = torch.tril(torch.ones(batch_length, batch_length)).numpy().astype(bool)
        lower_tri_values = head_mask[lower_tri]
        total_positions = len(lower_tri_values)
        masked_positions = np.sum(lower_tri_values == float('-inf'))
        actual_percent = masked_positions / total_positions
        
        print(f"  Expected mask %: {percent*100:.1f}%, Actual: {actual_percent*100:.1f}%")

if __name__ == "__main__":
    print("Mask Visualization Test Suite")
    print("=" * 40)
    
    # Test basic functionality
    success = test_basic_functionality()
    
    if success:
        # Test mask properties
        test_mask_properties()
        
        print("\n🎉 All tests completed successfully!")
        print("\nYou can now use the visualization functions in your code:")
        print("```python")
        print("from mask_visualization import visualize_multi_head_masks")
        print("fig = visualize_multi_head_masks(batch_length=12, mask_percents=[0.0, 0.25, 0.5, 0.75])")
        print("fig.show()  # or fig.savefig('output.png')")
        print("```")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)
