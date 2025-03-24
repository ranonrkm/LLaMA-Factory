#!/bin/bash

OUTPUT_DIR=/sensei-fs/users/xuhuang/rsadhukh/LLaMA-Factory
export ALLOW_EXTRA_ARGS=1
topk=$1
local=$2
ctx_len=$3
topk_iter=$4

llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --trust_remote_code true \
    --flash_attn fa2 \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --sparse_training \
    --sparse_attn_topk 256 \
    --sparse_attn_local ${local} \
    --sparse_attn_sink 4 \
    --sparsity_update_interval ${topk_iter} \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --dataset open_r1_math \
    --template qwen \
    --cutoff_len $ctx_len \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir ${OUTPUT_DIR}/saves/qwen2.5-3b/full/sft_sparse_ctx${ctx_len}_local${local}_top${topk}_iter${topk_iter} \
    --logging_steps 10 \
    --save_steps 1000 \
    --plot_loss \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --enable_liger_kernel \
    --learning_rate 5.0e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 5.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --save_total_limit 1 \
    --resume_from_checkpoint ${OUTPUT_DIR}/saves/qwen2.5-3b/full/sft_sparse_ctx${ctx_len}_local${local}_top${topk}_iter${topk_iter}/checkpoint-6000 \
    --push_to_hub --export_hub_model_id Rano23/OpenR1-qwen-3b-sft_sparse_ctx${ctx_len}_local${local}_top${topk}_iter${topk_iter}