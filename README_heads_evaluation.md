# Heads Experiments Evaluation Guide

This guide explains how to use the evaluation scripts for diversified and specialized heads experiments.

## Overview

Two evaluation scripts are available for evaluating the heads experiments:

1. **`eval_heads_experiments.py`** - Dedicated script for heads experiments with detailed configuration
2. **`eval_all_models_python.py`** - Updated existing script that now includes heads experiments

## Quick Start

### Using the Dedicated Heads Script

```bash
# Evaluate both diversified and specialized heads on MsPacman with 5 seeds
python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --num_seeds 5

# Quick test with fewer episodes and seeds
python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --num_seeds 3 --episodes 10

# Evaluate on Pong with custom seeds
python eval_heads_experiments.py --env_name ALE/Pong-v5 --seeds 42 43 44 45 46
```

### Using the Updated All-Models Script

```bash
# Evaluate all models including heads experiments
python eval_all_models_python.py --env_name ALE/MsPacman-v5

# Evaluate only heads experiments
python eval_all_models_python.py --env_name ALE/MsPacman-v5 --heads_only

# Evaluate all original masking models (exclude heads experiments)
python eval_all_models_python.py --env_name ALE/MsPacman-v5 --no_heads_experiments

# Evaluate on Pong environment
python eval_all_models_python.py --env_name ALE/Pong-v5 --heads_only
```

## Model Configurations

### Diversified Heads
- **Config**: `config_files/STORM_diversified_heads.yaml`
- **Masking Pattern**: `[0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]`
- **Strategy**: Even distribution across temporal scales for maximum diversity

### Specialized Heads
- **Config**: `config_files/STORM_specialized_heads.yaml`  
- **Masking Pattern**: `[0.0, 0.08, 0.15, 0.08, 0.25, 0.15, 0.35, 0.25]`
- **Strategy**: Functional specialization with grouped temporal roles

## Expected Checkpoint Structure

The evaluation scripts expect trained models to be stored in the following directory structure:

```
ckpt/
├── mspacman-diversified-sequential-50k-seed42/
│   ├── world_model_*.pth
│   └── agent_*.pth
├── mspacman-specialized-sequential-50k-seed42/
│   ├── world_model_*.pth
│   └── agent_*.pth
├── pong-diversified-sequential-50k-seed42/
│   ├── world_model_*.pth
│   └── agent_*.pth
└── pong-specialized-sequential-50k-seed42/
    ├── world_model_*.pth
    └── agent_*.pth
```

## Training the Models

To train the models for evaluation, use:

```bash
# Train diversified heads on MsPacman
python train.py \
    -n "mspacman-diversified-sequential-50k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_diversified_heads.yaml" \
    -env_name "ALE/MsPacman-v5" \
    -trajectory_path "D_TRAJ/MsPacman.pkl"

# Train specialized heads on MsPacman  
python train.py \
    -n "mspacman-specialized-sequential-50k-seed42" \
    -seed 42 \
    -config_path "config_files/STORM_specialized_heads.yaml" \
    -env_name "ALE/MsPacman-v5" \
    -trajectory_path "D_TRAJ/MsPacman.pkl"
```

## Output Files

Both evaluation scripts generate:

### JSON Files (Detailed Results)
- `eval_result/heads_experiments_mspacman_YYYYMMDD_HHMMSS.json`
- Contains complete evaluation data, individual rewards, confidence intervals

### CSV Files (Summary)
- `eval_result/heads_experiments_summary_mspacman_YYYYMMDD_HHMMSS.csv`
- Contains tabular summary for easy analysis and plotting

## Evaluation Parameters

### Default Settings
- **Episodes per evaluation**: 20
- **Evaluation seeds**: [1001, 1002, 1003, 1004, 1005]
- **Number of environments**: 5 (parallel evaluation)
- **Timeout**: 600 seconds per evaluation

### Customization Options
- `--num_seeds`: Number of random seeds to evaluate
- `--seeds`: Specific seed values to use
- `--eval_seeds`: Specific evaluation seeds (for reproducibility)
- `--episodes`: Number of episodes per evaluation run
- `--timeout`: Timeout in seconds for each evaluation

## Masking Strategy Logic

See `masking_strategy_explanation.md` for detailed explanation of:
- Why specialized heads use paired masking percentages
- How functional groups are organized
- Comparison with diversified heads approach
- Expected performance characteristics

## Statistical Analysis

The evaluation scripts provide:
- **Mean ± Standard Deviation** for each experiment
- **95% Confidence Intervals** for statistical significance testing
- **Success Rate** (percentage of successful evaluations)
- **Overlap Analysis** to detect statistically significant differences

## Troubleshooting

### Common Issues

1. **Missing Checkpoints**
   ```
   ❌ ERROR: Checkpoint directory not found: ckpt/mspacman-diversified-sequential-50k-seed42
   ```
   **Solution**: Train the models first using the training commands above.

2. **Missing Trajectory Files**
   ```
   ❌ Error: Trajectory file not found: D_TRAJ/MsPacman.pkl
   ```
   **Solution**: Ensure trajectory files exist in the D_TRAJ directory.

3. **Evaluation Timeout**
   ```
   ERROR: Evaluation timed out after 600s
   ```
   **Solution**: Increase timeout with `--timeout 900` or check system performance.

### Debug Mode

For debugging failed evaluations, check the individual error messages in the output or run with a single seed:

```bash
python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --seeds 42 --episodes 5
```

## Performance Expectations

### Typical Evaluation Times
- **Single evaluation**: 30-120 seconds (20 episodes)
- **Complete heads comparison**: 5-15 minutes (2 models × 5 seeds)
- **Full model suite**: 15-45 minutes (7 models × 5 seeds)

### Hardware Requirements
- **GPU**: Recommended for faster evaluation
- **RAM**: 8GB+ recommended for parallel environments
- **Storage**: ~500MB per trained model checkpoint
