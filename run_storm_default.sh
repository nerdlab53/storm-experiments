#!/bin/bash

set -euo pipefail

# Default arguments (override via env or flags)
SEED=${SEED:-1}
ENV_NAME=${ENV_NAME:-"ALE/Pong-v5"}
CONFIG_PATH=${CONFIG_PATH:-"config_files/STORM.yaml"}
TRAJ_PATH=${TRAJ_PATH:-"D_TRAJ/Pong.pkl"}
RUN_NAME=${RUN_NAME:-"${ENV_NAME##*/}-storm-default-seed${SEED}"}

# Optional wandb (leave empty to skip)
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_NAME=${WANDB_NAME:-}
WANDB_MODE=${WANDB_MODE:-}
WANDB_TAGS=${WANDB_TAGS:-}

cd "$(dirname "$0")"

CMD=(python -u train.py \
  -n "$RUN_NAME" \
  -seed "$SEED" \
  -config_path "$CONFIG_PATH" \
  -env_name "$ENV_NAME" \
  -trajectory_path "$TRAJ_PATH" \
  --world_model_impl default)

if [[ -n "$WANDB_PROJECT" ]]; then
  CMD+=(--wandb_project "$WANDB_PROJECT")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_GROUP" ]]; then
  CMD+=(--wandb_group "$WANDB_GROUP")
fi
if [[ -n "$WANDB_NAME" ]]; then
  CMD+=(--wandb_name "$WANDB_NAME")
fi
if [[ -n "$WANDB_MODE" ]]; then
  CMD+=(--wandb_mode "$WANDB_MODE")
fi
if [[ -n "$WANDB_TAGS" ]]; then
  # Split space-separated tags into array
  read -r -a TAG_ARR <<< "$WANDB_TAGS"
  CMD+=(--wandb_tags "${TAG_ARR[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


