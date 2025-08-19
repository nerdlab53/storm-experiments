# Masking Strategy Logic: Specialized vs Diversified Heads

## Overview

This document explains the strategic reasoning behind the masking percentage choices in the specialized heads experiment, and how it compares to the diversified heads approach.

## Background: Attention Head Masking in STORM

In the STORM (Structured Transformer for Reinforcement Learning) architecture, attention heads can be selectively masked to control how much historical context they can access. The masking percentages determine what fraction of the oldest tokens in the context are masked (made unavailable) to each attention head.

### Key Concepts:
- **0% masking**: Head has access to full context history
- **25% masking**: Head can only see the most recent 75% of context
- **95% masking**: Head can only see the most recent 5% of context (near-immediate focus)

## Specialized Heads Strategy

### Masking Pattern: `[0.0, 0.08, 0.15, 0.08, 0.25, 0.15, 0.35, 0.25]`

The specialized heads approach implements a **functional specialization strategy** where attention heads are organized into pairs with complementary masking levels, creating distinct functional groups.

### Functional Groups:

#### 1. **Strategic Planning Group (Heads 1-2: 0%, 8%)**
```
Heads: [0.0, 0.08]
Purpose: Long-term strategic planning & global pattern recognition
```
- **Head 1 (0% masking)**: Full context access for strategic planning
  - Can see entire episode history
  - Responsible for learning long-term dependencies
  - Captures global game state patterns and strategies

- **Head 2 (8% masking)**: Near-full context with slight temporal focus
  - Access to 92% of recent context
  - Provides strategic planning with slight bias toward recent events
  - Acts as a "strategic advisor" with mild recency preference

#### 2. **Tactical Decision Group (Heads 3-4: 15%, 8%)**
```
Heads: [0.15, 0.08]
Purpose: Medium-term tactical decisions and pattern adaptation
```
- **Head 3 (15% masking)**: Medium-term tactical planning
  - Access to 85% of recent context
  - Balances between strategy and immediate tactics
  - Learns medium-term patterns and adaptations

- **Head 4 (8% masking)**: Strategic-tactical bridge
  - Same as Head 2, providing consistency in strategic elements
  - Creates redundancy in strategic planning
  - Ensures robust strategic representation

#### 3. **Action Selection Group (Heads 5-6: 25%, 15%)**
```
Heads: [0.25, 0.15]
Purpose: Short-term action selection and immediate planning
```
- **Head 5 (25% masking)**: Short-term action planning
  - Access to 75% of recent context
  - Focuses on immediate action sequences
  - Balances current situation with recent history

- **Head 6 (15% masking)**: Tactical-action bridge
  - Same as Head 3, providing tactical consistency
  - Links medium-term tactics with immediate actions
  - Ensures smooth tactical-to-action transitions

#### 4. **Reflexive Response Group (Heads 7-8: 35%, 25%)**
```
Heads: [0.35, 0.25]
Purpose: Immediate reflexive responses and crisis handling
```
- **Head 7 (35% masking)**: Immediate response specialization
  - Access to 65% of recent context
  - Specialized for quick, reactive decisions
  - Handles time-critical situations and reflexes

- **Head 8 (25% masking)**: Action-reflex bridge
  - Same as Head 5, providing action consistency
  - Links planned actions with reflexive responses
  - Ensures coherent action execution

### Design Principles:

1. **Paired Redundancy**: Each masking level appears twice, creating robust representations at each temporal scale
2. **Graduated Specialization**: Smooth transition from strategic (0%) to reflexive (35%) focus
3. **Functional Hierarchy**: Clear division of labor from planning to execution
4. **Bridging Elements**: Repeated masking levels create bridges between functional groups

## Diversified Heads Strategy

### Masking Pattern: `[0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]`

The diversified heads approach uses **evenly distributed masking** across the full spectrum of temporal scales.

### Characteristics:
- **Uniform Distribution**: Even spacing from 0% to 95% masking
- **Broad Coverage**: Each head specializes in a different temporal scale
- **No Redundancy**: Each masking level is unique
- **Maximum Diversity**: Covers the widest range of temporal perspectives

## Comparative Analysis

| Aspect | Specialized Heads | Diversified Heads |
|--------|------------------|-------------------|
| **Philosophy** | Functional specialization with redundancy | Maximum temporal diversity |
| **Redundancy** | High (paired heads) | None (unique levels) |
| **Temporal Range** | 0% - 35% (moderate) | 0% - 95% (full spectrum) |
| **Functional Groups** | 4 clear groups | Individual specialists |
| **Robustness** | High (redundant representations) | Moderate (single points of failure) |
| **Specialization Depth** | Deep within groups | Broad across spectrum |

## Theoretical Advantages

### Specialized Heads Advantages:
1. **Robustness**: Redundant heads prevent single points of failure
2. **Functional Clarity**: Clear division of cognitive labor
3. **Smooth Transitions**: Bridging heads ensure coherent decision flow
4. **Focused Learning**: Paired heads can learn complementary aspects of each temporal scale

### Diversified Heads Advantages:
1. **Maximum Coverage**: Access to full spectrum of temporal scales
2. **Unique Perspectives**: Each head offers a distinct viewpoint
3. **Flexibility**: No predetermined functional constraints
4. **Extreme Specialization**: Includes very high masking (85%, 95%) for immediate-only focus

## Expected Performance Characteristics

### Specialized Heads:
- **Strengths**: Consistent performance, robust decision-making, good strategic planning
- **Potential Weaknesses**: May miss very short-term patterns, less extreme specialization

### Diversified Heads:
- **Strengths**: Excellent at diverse temporal patterns, flexible adaptation
- **Potential Weaknesses**: Potential for inconsistent performance, single points of failure

## Implementation Details

### Masking Mechanism:
- **Type**: Oldest-first masking (removes oldest tokens first)
- **Application**: Hard masking (complete unavailability of masked tokens)
- **Consistency**: Same masking pattern applied consistently across training

### Training Implications:
- **Specialization**: Each head learns to specialize in its assigned temporal range
- **Interaction**: Heads must learn to combine their specialized knowledge
- **Adaptation**: Network adapts to work within the constraints of each masking pattern

## Evaluation Strategy

The evaluation script (`eval_heads_experiments.py`) tests both strategies across multiple seeds to determine:

1. **Performance**: Which strategy achieves higher average rewards
2. **Consistency**: Which strategy has lower variance across seeds
3. **Robustness**: Which strategy performs better under different random initializations
4. **Statistical Significance**: Whether performance differences are statistically meaningful

This systematic comparison will reveal the practical benefits of functional specialization versus maximum temporal diversity in attention head design for reinforcement learning tasks.
