env_name=Pong
python -u train.py \
    -n "${env_name}-life_done-wm_2L512D8H-25k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM_25k.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" 