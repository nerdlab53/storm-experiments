#!/usr/bin/env python3
"""
Quick test to verify the visualization fix works.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mask_visualization import visualize_multi_head_masks, quick_demo
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_single_head():
    """Test with just 1 head (edge case that was causing issues)."""
    print("Testing single head (edge case)...")
    
    try:
        fig = visualize_multi_head_masks(
            batch_length=6,
            mask_percents=[0.5],  # Just one head
            flag=False,
            soft=True,
            device='cpu',
            figsize=(8, 6)
        )
        fig.savefig('test_single_head_fix.png', dpi=150, bbox_inches='tight')
        print("✓ Single head test passed")
        return True
    except Exception as e:
        print(f"✗ Single head test failed: {e}")
        return False

def test_multi_head():
    """Test with multiple heads."""
    print("Testing multiple heads...")
    
    try:
        fig = visualize_multi_head_masks(
            batch_length=8,
            mask_percents=[0.0, 0.25, 0.5, 0.75],
            flag=False,
            soft=True,
            device='cpu',
            figsize=(16, 8)
        )
        fig.savefig('test_multi_head_fix.png', dpi=150, bbox_inches='tight')
        print("✓ Multi-head test passed")
        return True
    except Exception as e:
        print(f"✗ Multi-head test failed: {e}")
        return False

def test_quick_demo():
    """Test the quick demo function."""
    print("Testing quick demo...")
    
    try:
        success = quick_demo(batch_length=6, save_plots=True)
        if success:
            print("✓ Quick demo test passed")
            return True
        else:
            print("✗ Quick demo returned False")
            return False
    except Exception as e:
        print(f"✗ Quick demo test failed: {e}")
        return False

if __name__ == "__main__":
    print("Quick Test Suite for Mask Visualization Fix")
    print("=" * 50)
    
    tests = [
        test_single_head,
        test_multi_head,
        test_quick_demo
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fix is working.")
        print("\nGenerated test files:")
        print("  - test_single_head_fix.png")
        print("  - test_multi_head_fix.png")
        print("  - demo_multi_head.png")
        print("  - demo_strategies.png")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        sys.exit(1)
