#!/bin/bash
#SBATCH --job-name=llama3_lora_sft
#SBATCH --partition=preempt
#SBATCH --nodes=2                    # Request 2 nodes
#SBATCH --ntasks-per-node=1          # Run 1 task per node
#SBATCH --cpus-per-task=16           # Adjust based on your node CPU count
#SBATCH --gres=gpu:8                 # Adjust based on available GPUs per node
#SBATCH --mem=64G                    # Memory per node
#SBATCH --time=24:00:00              # Maximum job runtime
#SBATCH --output=logs/sft_%x_%j.out           # Standard output file
#SBATCH --error=logs/sft_%x_%j.err            # Standard error file

# Get the master node hostname
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Master node: $MASTER_ADDR"

# Set environment variables for distributed training
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export NNODES=$SLURM_NNODES
export FORCE_TORCHRUN=1

# Run training script based on node rank
srun --export=ALL bash -c 'NODE_RANK=$SLURM_PROCID llamafactory-cli train examples/train_lora/qwen7b_lora_sft_sparse.yaml'