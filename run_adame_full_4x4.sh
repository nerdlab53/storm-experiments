env_name=Breakout
python -u train.py \
    -n "${env_name}-adamae-full-4x4-learnedratio-wm_2L512D8H-50k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM_AdaMAE_Full_4x4.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    --use_adamae