# Changes Documentation

## Novelty Detection

**Location**: `sub_models/novelty_detector.py`
- Main implementation in `WorldModelNoveltyWrapper` class
- Uses KL divergence and Expected Information Gain for detection
- Integrated into training pipeline in `train.py` (lines 138-154)
- Configuration available in config files under `Models.NoveltyDetection`
- Detection logs saved to `detection_logs/` directory

**Config Files**: 
- `config_files/STORM.yaml` - Full novelty detection enabled | standard config.
- `config_files/STORM_novelty_test.yaml` - Testing configuration | breaks training right now, checking.
- I recommend using standard config.

## Progressive Masking

**Location**: `sub_models/attention_blocks.py`
- Core implementation in `get_progressive_causal_mask()` function
- Progressive attention mechanisms in `ScaledDotProductAttentionProgressive`
- Multi-head attention support in `MultiHeadAttentionProgressive`

**Integration**: `sub_models/world_models.py`
- WorldModel class uses progressive masking when `use_progressive_masking=True`
- Applied in `calc_last_dist_feat()` and `update()` methods

**Transformer**: `sub_models/transformer_model.py`
- `StochasticTransformerKVCacheProgressive` class
- KV cache support with progressive masking

**Configuration**: Set via `UseProgressiveMasking` parameter in config files

## Token Compression

**Status**: In Progress