#!/bin/bash

# Example usage script for heads experiments evaluation
# This script demonstrates how to run the evaluation for both MsPacman and Pong environments

echo "🎯 Heads Experiments Evaluation - Example Usage"
echo "=============================================="
echo

# Check if the evaluation script exists
if [ ! -f "eval_heads_experiments.py" ]; then
    echo "❌ Error: eval_heads_experiments.py not found!"
    echo "Please make sure you're in the storm-experiments directory."
    exit 1
fi

# Check for required trajectory files
echo "📁 Checking trajectory files..."
if [ ! -f "D_TRAJ/MsPacman.pkl" ]; then
    echo "⚠️  Warning: D_TRAJ/MsPacman.pkl not found - MsPacman evaluation will fail"
fi
if [ ! -f "D_TRAJ/Pong.pkl" ]; then
    echo "⚠️  Warning: D_TRAJ/Pong.pkl not found - Pong evaluation will fail"
fi
echo

# Example 1: Quick test with MsPacman (3 seeds, 10 episodes)
echo "🚀 Example 1: Quick test with MsPacman"
echo "Command: python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --num_seeds 3 --episodes 10"
echo "This will evaluate both diversified and specialized heads with seeds [42, 43, 44]"
echo

# Example 2: Full evaluation with Pong (5 seeds, 20 episodes)
echo "🚀 Example 2: Full evaluation with Pong"
echo "Command: python eval_heads_experiments.py --env_name ALE/Pong-v5 --num_seeds 5"
echo "This will evaluate with seeds [42, 43, 44, 45, 46] using 20 episodes each"
echo

# Example 3: Custom seeds evaluation
echo "🚀 Example 3: Custom seeds evaluation"
echo "Command: python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --seeds 100 101 102 --eval_seeds 3000 3001 3002"
echo "This will evaluate with specific training and evaluation seeds"
echo

# Example 4: Show available options
echo "🚀 Example 4: Show all available options"
echo "Command: python eval_heads_experiments.py --help"
echo

# Ask user which example to run
echo "💭 Which example would you like to run?"
echo "1) Quick test with MsPacman (3 seeds, 10 episodes)"
echo "2) Full evaluation with Pong (5 seeds, 20 episodes)" 
echo "3) Custom seeds with MsPacman"
echo "4) Show help"
echo "5) Exit"
echo

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Running Example 1..."
        python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --num_seeds 3 --episodes 10
        ;;
    2)
        echo "Running Example 2..."
        python eval_heads_experiments.py --env_name ALE/Pong-v5 --num_seeds 5
        ;;
    3)
        echo "Running Example 3..."
        python eval_heads_experiments.py --env_name ALE/MsPacman-v5 --seeds 100 101 102 --eval_seeds 3000 3001 3002
        ;;
    4)
        echo "Showing help..."
        python eval_heads_experiments.py --help
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Showing help instead..."
        python eval_heads_experiments.py --help
        ;;
esac

echo
echo "✅ Example completed!"
echo "📄 Check the eval_result/ directory for output files."
echo "📊 JSON files contain detailed results, CSV files contain summaries."
