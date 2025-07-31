#!/usr/bin/env python3
"""
Script to analyze model parameters for STORM World Model architecture.
Usage: python analyze_model_params.py -config_path config_files/STORM.yaml -env_name "ALE/Pong-v5"
"""

import argparse
import torch
import torch.nn as nn
from utils import load_config
import train
import agents
from sub_models.world_models import WorldModel, EncoderBN, DecoderBN, DistHead, RewardDecoder, TerminationDecoder
from sub_models.transformer_model import StochasticTransformerKVCacheProgressive
from sub_models.attention_blocks import MultiHeadAttentionProgressive, AttentionBlockKVCacheProgressive
import gymnasium


def count_parameters(model, only_trainable=True):
    """Count parameters in a model"""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def analyze_module_params(module, name="Module", indent=0):
    """Recursively analyze parameters in a module"""
    prefix = "  " * indent
    total_params = count_parameters(module)
    trainable_params = count_parameters(module, only_trainable=True)
    
    print(f"{prefix}{name}:")
    print(f"{prefix}  Total params: {total_params:,}")
    print(f"{prefix}  Trainable params: {trainable_params:,}")
    
    # Analyze submodules
    submodule_details = {}
    for subname, submodule in module.named_children():
        if len(list(submodule.parameters())) > 0:  # Only if it has parameters
            subparams = count_parameters(submodule)
            submodule_details[subname] = subparams
            print(f"{prefix}    {subname}: {subparams:,} params")
    
    return total_params, trainable_params, submodule_details


def analyze_world_model_components(world_model, config):
    """Detailed analysis of WorldModel components"""
    print("\n" + "="*60)
    print("WORLD MODEL COMPONENT BREAKDOWN")
    print("="*60)
    
    total_params = 0
    component_params = {}
    
    # Encoder (Tokenizer)
    print("\n1. ENCODER (Frame Tokenizer):")
    print("   Converts 64x64x3 images to 4x4x512 feature maps")
    encoder_params, _, _ = analyze_module_params(world_model.encoder, "EncoderBN", indent=1)
    component_params['encoder'] = encoder_params
    total_params += encoder_params
    
    # Distribution Head (VQVAE-style)
    print("\n2. DISTRIBUTION HEAD (Stochastic Tokenizer):")
    print("   Converts features to discrete latent tokens (32x32 categorical)")
    dist_params, _, _ = analyze_module_params(world_model.dist_head, "DistHead", indent=1)
    component_params['dist_head'] = dist_params
    total_params += dist_params
    
    # Transformer
    print("\n3. TRANSFORMER (Sequence Model):")
    print(f"   {config.Models.WorldModel.TransformerNumLayers} layers, {config.Models.WorldModel.TransformerNumHeads} heads, {config.Models.WorldModel.TransformerHiddenDim} hidden dim")
    print(f"   Max sequence length: {config.Models.WorldModel.TransformerMaxLength}")
    transformer_params, _, transformer_details = analyze_module_params(world_model.storm_transformer, "StochasticTransformerKVCacheProgressive", indent=1)
    component_params['transformer'] = transformer_params
    total_params += transformer_params
    
    # Image Decoder
    print("\n4. IMAGE DECODER:")
    print("   Reconstructs images from latent tokens")
    decoder_params, _, _ = analyze_module_params(world_model.image_decoder, "DecoderBN", indent=1)
    component_params['image_decoder'] = decoder_params
    total_params += decoder_params
    
    # Reward Decoder
    print("\n5. REWARD DECODER:")
    print("   Predicts rewards from latent features")
    reward_params, _, _ = analyze_module_params(world_model.reward_decoder, "RewardDecoder", indent=1)
    component_params['reward_decoder'] = reward_params
    total_params += reward_params
    
    # Termination Decoder
    print("\n6. TERMINATION DECODER:")
    print("   Predicts episode termination from latent features")
    term_params, _, _ = analyze_module_params(world_model.termination_decoder, "TerminationDecoder", indent=1)
    component_params['termination_decoder'] = term_params
    total_params += term_params
    
    print(f"\nTotal World Model Parameters: {total_params:,}")
    return total_params, component_params


def analyze_agent_components(agent, config):
    """Detailed analysis of Actor-Critic Agent"""
    print("\n" + "="*60)
    print("ACTOR-CRITIC AGENT BREAKDOWN")
    print("="*60)
    
    total_params = 0
    component_params = {}
    
    # Calculate input dimension
    feat_dim = 32*32 + config.Models.WorldModel.TransformerHiddenDim  # latent + transformer features
    
    print(f"\nInput dimension: {feat_dim} (32x32 latent + {config.Models.WorldModel.TransformerHiddenDim} transformer features)")
    print(f"Hidden dimension: {config.Models.Agent.HiddenDim}")
    print(f"Number of layers: {config.Models.Agent.NumLayers}")
    
    # Actor
    print("\n1. ACTOR (Policy Network):")
    print("   Maps state features to action probabilities")
    actor_params, _, _ = analyze_module_params(agent.actor, "Actor", indent=1)
    component_params['actor'] = actor_params
    total_params += actor_params
    
    # Critic
    print("\n2. CRITIC (Value Network):")
    print("   Maps state features to value estimates (255 classes for symlog encoding)")
    critic_params, _, _ = analyze_module_params(agent.critic, "Critic", indent=1)
    component_params['critic'] = critic_params
    total_params += critic_params
    
    # Slow Critic
    print("\n3. SLOW CRITIC (Target Network):")
    print("   EMA copy of critic for stable training")
    slow_critic_params, _, _ = analyze_module_params(agent.slow_critic, "SlowCritic", indent=1)
    component_params['slow_critic'] = slow_critic_params
    total_params += slow_critic_params
    
    print(f"\nTotal Agent Parameters: {total_params:,}")
    return total_params, component_params


def analyze_transformer_details(transformer, config):
    """Detailed breakdown of transformer architecture"""
    print("\n" + "="*50)
    print("TRANSFORMER DETAILED BREAKDOWN")
    print("="*50)
    
    # Input embedding
    input_dim = 32*32 + 6  # stoch_flattened_dim + action_dim (Pong has 6 actions)
    feat_dim = config.Models.WorldModel.TransformerHiddenDim
    
    print(f"\nInput Processing:")
    print(f"  Input tokens: {32*32} (flattened stochastic latent) + {6} (one-hot action)")
    print(f"  Total input dim: {input_dim}")
    print(f"  Feature dim: {feat_dim}")
    
    # Stem
    stem_params = count_parameters(transformer.stem)
    print(f"\nStem (Input Projection): {stem_params:,} params")
    
    # Position encoding
    pos_params = count_parameters(transformer.position_encoding)
    print(f"Position Encoding: {pos_params:,} params")
    
    # Attention layers
    print(f"\nAttention Layers ({config.Models.WorldModel.TransformerNumLayers} layers):")
    total_attention_params = 0
    for i, layer in enumerate(transformer.layer_stack):
        layer_params = count_parameters(layer)
        total_attention_params += layer_params
        print(f"  Layer {i+1}: {layer_params:,} params")
        
        # Break down attention layer
        if hasattr(layer, 'slf_attn'):
            attn_params = count_parameters(layer.slf_attn)
            print(f"    Self-attention: {attn_params:,} params")
        if hasattr(layer, 'pos_ffn'):
            ffn_params = count_parameters(layer.pos_ffn)
            print(f"    Feed-forward: {ffn_params:,} params")
    
    # Output head
    head_params = count_parameters(transformer.head)
    print(f"\nOutput Head: {head_params:,} params")
    
    total = stem_params + pos_params + total_attention_params + head_params
    print(f"\nTransformer Total: {total:,} params")


def main():
    parser = argparse.ArgumentParser(description="Analyze STORM model parameters")
    parser.add_argument("-config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("-env_name", type=str, required=True, help="Environment name")
    parser.add_argument("--detailed", action="store_true", help="Show detailed transformer breakdown")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config_path)
    
    # Get action dimension
    dummy_env = train.build_single_env(args.env_name, config.BasicSettings.ImageSize, seed=0)
    action_dim = dummy_env.action_space.n
    dummy_env.close()
    
    print("="*80)
    print("STORM MODEL PARAMETER ANALYSIS")
    print("="*80)
    print(f"Config: {args.config_path}")
    print(f"Environment: {args.env_name}")
    print(f"Action dimension: {action_dim}")
    print(f"Image size: {config.BasicSettings.ImageSize}x{config.BasicSettings.ImageSize}")
    
    # Build models
    print("\nBuilding models...")
    world_model = train.build_world_model(config, action_dim)
    agent = train.build_agent(config, action_dim)
    
    # Remove novelty wrapper if present to analyze core world model
    if hasattr(world_model, 'world_model'):
        core_world_model = world_model.world_model
        print("Note: Analyzing core WorldModel (novelty detection wrapper detected)")
    else:
        core_world_model = world_model
    
    # Analyze components
    wm_total, wm_components = analyze_world_model_components(core_world_model, config)
    agent_total, agent_components = analyze_agent_components(agent, config)
    
    # Detailed transformer analysis
    if args.detailed:
        analyze_transformer_details(core_world_model.storm_transformer, config)
    
    # Summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    grand_total = wm_total + agent_total
    
    print(f"\nWorld Model:     {wm_total:,} parameters ({wm_total/grand_total*100:.1f}%)")
    print(f"Actor-Critic:    {agent_total:,} parameters ({agent_total/grand_total*100:.1f}%)")
    print(f"{'':>16}{'='*30}")
    print(f"Grand Total:     {grand_total:,} parameters")
    
    # Memory estimation (rough)
    param_size_mb = grand_total * 4 / (1024**2)  # Assuming float32
    print(f"\nEstimated memory (FP32): {param_size_mb:.1f} MB")
    print(f"Estimated memory (FP16): {param_size_mb/2:.1f} MB")
    
    # Component breakdown
    print(f"\nWorld Model breakdown:")
    for comp, params in wm_components.items():
        print(f"  {comp:>20}: {params:,} ({params/wm_total*100:.1f}%)")
    
    print(f"\nAgent breakdown:")
    for comp, params in agent_components.items():
        print(f"  {comp:>20}: {params:,} ({params/agent_total*100:.1f}%)")


if __name__ == "__main__":
    main() 