#!/bin/bash

CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python ./sft.py \
    --model_name="mistralai/Mistral-7B-v0.1" \
    --output_dir="./models/sft_mistral_tldr" \
    --dataset_name="CarperAI/openai_summarize_tldr" \
    --max_steps=500 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="sft" \
    --report_to="wandb" \
    # --model_name="meta-llama/Llama-2-7b-hf" \
    # --model_name="mistralai/Mistral-7B-v0.1" \
    # --dataset_name="EdinburghNLP/xsum" \
    # --dataset_name="CarperAI/openai_summarize_tldr" \
    
