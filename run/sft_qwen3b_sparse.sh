#!/bin/bash

export ALLOW_EXTRA_ARGS=1
topk=$1
local=$2
ctx_len=$3
topk_iter=$4

llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --trust_remote_code true \
    --flash_attn fa2 \
    --stage sft \
    --do_train true \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --finetuning_type layerwise \
    --sparse_training \
    --sparse_attn_method ivf \
    --sparse_attn_topk ${topk} \
    --sparse_attn_local ${local} \
    --sparse_attn_sink 4 \
    --sparsity_update_interval ${topk_iter} \
    --dataset open_r1_math \
    --template qwen \
    --cutoff_len $ctx_len \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --output_dir saves/qwen2.5-3b/temp \
    --logging_steps 2 \
    --save_steps 1000 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --enable_liger_kernel true \
    --learning_rate 5.0e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 5.0 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --save_total_limit 1 #\
    # --resume_from_checkpoint 
    # --deepspeed examples/deepspeed/ds_z3_config.json \
    