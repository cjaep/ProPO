## Installation 
Our code is based on Huggingface's `transformers>=4.35.0`.

```bash
conda create -n ProPO python=3.8 -y

conda activate ProPO
pip install torch torchvision torchaudio

pip install datasets evaluate rouge-score nltk
```



## Supervised Fine-Tuning
Please refer to the code example below for instructions on how to run the code.

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


## Evaluation

To perform evaluation, you need to install the metrics.
- [AlignScore](https://github.com/yuh-zha/AlignScore)
- [BARTScore](https://github.com/neulab/BARTScore)


```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python evaluation.py \
    --input_file "results/{OUTPUT_FILE}.json" \
    --output_file "results/evaluation.json" \
    --batch_size 16 \
    --alignscore_ckpt "{PATH_TO_ALIGNSCORECKPT}" \
```


# Citation
```
@inproceedings{
chae2024mitigating,
title={Mitigating Hallucination in Abstractive Summarization with Domain-Conditional Mutual Information},
author={Kyubyung Chae and Jaepill choi and Yohan Jo and Taesup Kim},
booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=N5gW9kxJ7Z}
}
```
