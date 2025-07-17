#!/bin/bash

# Default values
env_name=Pong
device=auto
run_name="${env_name}-life_done-wm_2L512D8H-25k-seed1"
config_path="config_files/STORM_25k.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            device="$2"
            shift 2
            ;;
        --env_name)
            env_name="$2"
            shift 2
            ;;
        --run_name)
            run_name="$2"
            shift 2
            ;;
        --config_path)
            config_path="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --device DEVICE        Device to use (auto, cpu, cuda, mps). Default: auto"
            echo "  --env_name ENV         Environment name. Default: Pong"
            echo "  --run_name NAME        Run name. Default: Pong-life_done-wm_2L512D8H-25k-seed1"
            echo "  --config_path PATH     Config file path. Default: config_files/STORM_25k.yaml"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running evaluation with:"
echo "  Environment: $env_name"
echo "  Device: $device"
echo "  Run name: $run_name"
echo "  Config: $config_path"
echo

python -u eval.py \
    -env_name "ALE/${env_name}-v5" \
    -run_name "$run_name" \
    -config_path "$config_path" \
    -device "$device" 
