#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_7b_sparse_local1k_top256.out
#SBATCH --error=logs/sft_7b_sparse_local1k_top256.err

export ALLOW_EXTRA_ARGS=1
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --trust_remote_code \
  --stage sft \
  --do_train \
  --finetuning_type lora \
  --sparse_training \
  --sparse_attn_topk 256 \
  --local 512 \
  --sink 4 \
  --lora_rank 16 \
  --lora_target all \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset open_r1_math \
  --template qwen \
  --cutoff_len 16384 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --output_dir saves/qwen2.5-7b/lora/sft_sparse_ctx16k_local512_top256 \
  --logging_steps 10 \
  --save_steps 1000 \
  --plot_loss \
  --overwrite_output_dir \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --enable_liger_kernel \
  --learning_rate 1.0e-4 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --ddp_timeout 180000000 \
  --save_total_limit 1