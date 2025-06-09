#!/bin/sh

# 如果使用llama模型进行推理，将'use_lora'设置为'False'，'prompt_template'设置为'ori_template'。
# 如果使用默认的alpaca模型进行推理，将'use_lora'设置为'True'，'lora_weights'设置为'tloen/alpaca-lora-7b'，'prompt_template'设置为'alpaca'。
# 如果使用llama-med模型进行推理，下载LORA权重并将'lora_weights'设置为'./lora-llama-med'（或LORA权重的具体目录），'prompt_template'设置为'med_template'。
# 需要提供lora权重，以及推理数据，以及模板
python infer.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './lora-llama-med' \
    --use_lora True \
    --instruct_dir './data/infer.json' \
    --prompt_template 'med_template'
