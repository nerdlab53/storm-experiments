import argparse
import os
import subprocess
import sys


def build_base_cmd(args: argparse.Namespace, run_name: str, world_model_impl: str) -> list:
    cmd = [
        sys.executable, "-u", "train.py",
        "-n", run_name,
        "-seed", str(args.seed),
        "-config_path", args.config_path,
        "-env_name", args.env_name,
        "-trajectory_path", args.trajectory_path,
        "--world_model_impl", world_model_impl,
    ]

    # Wandb flags are optional
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
    if args.wandb_entity:
        cmd += ["--wandb_entity", args.wandb_entity]
    if args.wandb_group:
        cmd += ["--wandb_group", args.wandb_group]
    if args.wandb_name:
        cmd += ["--wandb_name", args.wandb_name]
    if args.wandb_mode:
        cmd += ["--wandb_mode", args.wandb_mode]
    if args.wandb_tags:
        cmd += ["--wandb_tags", *args.wandb_tags]

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run STORM default and adamae variants with optional wandb logging")
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-config_path", type=str, default="config_files/STORM.yaml")
    parser.add_argument("-env_name", type=str, default="ALE/Pong-v5")
    parser.add_argument("-trajectory_path", type=str, default="D_TRAJ/Pong.pkl")
    parser.add_argument("--run_prefix", type=str, default=None, help="Prefix for run names")
    # wandb options (forwarded to train.py)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_tags", nargs='*', default=None)

    args = parser.parse_args()

    # Derive env short name for run names
    env_short = args.env_name.split('/')[-1]
    prefix = args.run_prefix or env_short

    # Default STORM
    run_name_default = f"{prefix}-storm-default-seed{args.seed}"
    cmd_default = build_base_cmd(args, run_name_default, world_model_impl="default")
    print("Running:", " ".join(cmd_default))
    subprocess.run(cmd_default, check=True)

    # STORM with adamae world model
    run_name_adamae = f"{prefix}-storm-adamae-seed{args.seed}"
    # Ensure distinct wandb name if set
    if hasattr(args, 'wandb_name') and args.wandb_name:
        args.wandb_name = f"{args.wandb_name}-adamae"
    cmd_adamae = build_base_cmd(args, run_name_adamae, world_model_impl="adamae")
    print("Running:", " ".join(cmd_adamae))
    subprocess.run(cmd_adamae, check=True)


if __name__ == "__main__":
    main()


