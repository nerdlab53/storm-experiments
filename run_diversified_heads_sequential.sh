ENV_NAME="MsPacman"
GAME_ENV="ALE/MsPacman-v5"
TRAJECTORY_PATH="D_TRAJ/MsPacman.pkl"
SEED=42
EXPERIMENT_NAME="mspacman-diversified-sequential-50k-seed${SEED}"

echo "Starting training with diversified head masking (sequential)..."
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Max steps: 50,000"
echo "----------------------------------------"

python -u train.py \
    -n "${EXPERIMENT_NAME}" \
    -seed ${SEED} \
    -config_path "config_files/STORM_diversified_heads.yaml" \
    -env_name "${GAME_ENV}" \
    -trajectory_path "${TRAJECTORY_PATH}"

echo "Diversified heads (sequential) experiment completed!"
echo "Results saved in: runs/${EXPERIMENT_NAME}/"
