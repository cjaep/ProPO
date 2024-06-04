## Installation 
Our code is based on Huggingface's `transformers>=4.35.0`.

```bash
conda create -n ProPO python=3.8 -y

conda activate ProPO
pip install torch torchvision torchaudio
pip install datasets evaluate rouge-score nltk
```



## SFT
The implementation follows an example from https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts

```sh
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
```

## DPO
```sh
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python dpo.py \
    --rejected_file="${REJECTED_FILE}" \
    --chosen_file="${CHOSEN_FILE}" \
    --model_name_or_path="models/${BASE_MODEL}" \
    --output_dir="models/${TARGET_MODEL}" \
    --beta=0.5 \
    --learning_rate=1e-4 \
    --warmup_steps=150 \
    --max_length=2048 \
    --max_prompt_length=2000 \
    --num_train_epochs=1 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
```

## Summarization
```sh
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python summarization.py \
    --output_file="${FOLDER}/${DATE}_${TARGET_MODEL}_${SPLIT}_${DECODING_TYPE}" \
    --decoding_type="${DECODING_TYPE}" \
    --model="models/${TARGET_MODEL}/final_checkpoint" \
    --tokenizer_model="${TOKENIZER_MODEL}" \
    --batch_size=1 \
    --dataset="${DATASET}" \
    --split="${SPLIT}" \
```

## Evaluation

To perform evaluation, you need to install the metrics.
- [AlignScore](https://github.com/yuh-zha/AlignScore)
- [BARTScore](https://github.com/neulab/BARTScore)


```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python ../../dcpmi/evaluation.py \
    --input_file "${FOLDER}/${DATE}_${TARGET_MODEL}_${SPLIT}_${DECODING_TYPE}.json" \
    --output_file "${FOLDER}/evaluation.json" \
    --batch_size 16 \
    --alignscore_ckpt "{PATH_TO_ALIGNSCORECKPT}" \
```

