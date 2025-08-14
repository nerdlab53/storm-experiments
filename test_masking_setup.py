#!/usr/bin/env python3
"""
Test script to validate the attention head masking setup before training.
This script checks:
1. Configuration loading
2. WorldModel initialization with per-head masking
3. Masking function validation
4. Training readiness
"""

import torch
import yaml
import sys
import traceback
from utils import load_config
from sub_models.world_models import WorldModel

def test_config_loading():
    """Test that configurations load properly."""
    print("🔧 Testing configuration loading...")
    
    configs_to_test = [
        'config_files/STORM_diversified_heads.yaml',
        'config_files/STORM_specialized_heads.yaml'
    ]
    
    for config_path in configs_to_test:
        try:
            config = load_config(config_path)
            percents = getattr(config.Models.WorldModel, 'FixedMaskPercents', None)
            num_heads = config.Models.WorldModel.TransformerNumHeads
            
            if percents is None:
                print(f"❌ {config_path}: FixedMaskPercents not found")
                continue
                
            if len(percents) != num_heads:
                print(f"❌ {config_path}: Length mismatch - {len(percents)} percents for {num_heads} heads")
                continue
                
            if not all(0.0 <= p <= 1.0 for p in percents):
                print(f"❌ {config_path}: Invalid percentage values")
                continue
                
            print(f"✅ {config_path}: {percents}")
            
        except Exception as e:
            print(f"❌ {config_path}: {str(e)}")
    
    print()

def test_world_model_creation():
    """Test WorldModel creation with per-head masking."""
    print("🏗️  Testing WorldModel creation...")
    
    try:
        # Test with diversified heads configuration
        config = load_config('config_files/STORM_diversified_heads.yaml')
        
        world_model = WorldModel(
            in_channels=config.Models.WorldModel.InChannels,
            action_dim=6,  # Typical Atari action space
            transformer_max_length=config.Models.WorldModel.TransformerMaxLength,
            transformer_hidden_dim=config.Models.WorldModel.TransformerHiddenDim,
            transformer_num_layers=config.Models.WorldModel.TransformerNumLayers,
            transformer_num_heads=config.Models.WorldModel.TransformerNumHeads,
            use_progressive_masking=getattr(config.Models.WorldModel, 'UseProgressiveMasking', True),
            use_progressive_in_kv=getattr(config.Models.WorldModel, 'UseProgressiveInKVCache', False),
            use_mild_decay_in_kv=getattr(config.Models.WorldModel, 'UseMildDecayInKV', False),
            fixed_mask_percent=getattr(config.Models.WorldModel, 'FixedMaskPercent', 0.0),
            fixed_mask_percents=getattr(config.Models.WorldModel, 'FixedMaskPercents', None),
            use_random_mask=getattr(config.Models.WorldModel, 'UseRandomMask', False),
            use_soft_penalty=getattr(config.Models.WorldModel, 'UseSoftPenalty', True)
        )
        
        print(f"✅ WorldModel created successfully")
        print(f"   Masking percentages: {world_model.fixed_mask_percents}")
        print(f"   Number of heads: {world_model.num_heads}")
        
    except Exception as e:
        print(f"❌ WorldModel creation failed: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
    
    print()

def test_masking_functionality():
    """Test the masking function directly."""
    print("🎭 Testing masking functionality...")
    
    try:
        from sub_models.attention_blocks import get_per_head_fixed_mask_causal
        
        # Test parameters
        batch_length = 16
        mask_percents = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        device = torch.device('cpu')
        
        mask = get_per_head_fixed_mask_causal(
            batch_length=batch_length,
            mask_percents=mask_percents,
            flag=False,  # oldest-first masking
            soft=False,  # hard masking
            device=device
        )
        
        expected_shape = (1, len(mask_percents), batch_length, batch_length)
        if mask.shape == expected_shape:
            print(f"✅ Mask generation successful")
            print(f"   Shape: {mask.shape}")
            print(f"   Device: {mask.device}")
            
            # Check that different heads have different patterns
            unique_patterns = []
            for h in range(len(mask_percents)):
                pattern = mask[0, h].sum().item()
                unique_patterns.append(pattern)
            
            if len(set(unique_patterns)) > 1:
                print(f"✅ Different heads have different masking patterns")
            else:
                print(f"⚠️  All heads have similar patterns - check configuration")
        else:
            print(f"❌ Unexpected mask shape: {mask.shape}, expected: {expected_shape}")
            
    except Exception as e:
        print(f"❌ Masking test failed: {str(e)}")
    
    print()

def main():
    """Run all tests."""
    print("🚀 Testing Attention Head Masking Setup")
    print("=" * 50)
    
    test_config_loading()
    test_world_model_creation()
    test_masking_functionality()
    
    print("🎉 Setup validation complete!")
    print("\n💡 To start training with diversified head masking:")
    print("   python train.py --config_path config_files/STORM_diversified_heads.yaml --env_name BreakoutNoFrameskip-v4 --n experiment_diversified")
    print("\n💡 To start training with specialized head masking:")
    print("   python train.py --config_path config_files/STORM_specialized_heads.yaml --env_name BreakoutNoFrameskip-v4 --n experiment_specialized")

if __name__ == "__main__":
    main()
