#!/bin/bash

OUTPUT_DIR=/sensei-fs/users/xuhuang/rsadhukh/LLaMA-Factory
export ALLOW_EXTRA_ARGS=1
ctx_len=$1

llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --trust_remote_code true \
    --flash_attn fa2 \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --dataset open_r1_math \
    --template qwen \
    --cutoff_len $ctx_len \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --output_dir ${OUTPUT_DIR}/saves/qwen2.5-3b/full/sft_ctx${ctx_len} \
    --logging_steps 10 \
    --save_steps 1000 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --enable_liger_kernel true \
    --learning_rate 5.0e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 5.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --save_total_limit 1 \
    --push_to_hub --export_hub_model_id Rano23/OpenR1-qwen-3b-sft-ctx${ctx_len}