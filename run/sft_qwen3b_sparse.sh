#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_3b_sparse.out
#SBATCH --error=logs/sft_3b_sparse.err

export ALLOW_EXTRA_ARGS=1
llamafactory-cli train examples/train_full/qwen3b_sft_sparse.yaml