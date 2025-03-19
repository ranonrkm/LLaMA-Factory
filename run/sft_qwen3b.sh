#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_3b_16k.out
#SBATCH --error=logs/sft_3b_16k.err

llamafactory-cli train examples/train_full/qwen3b_sft.yaml