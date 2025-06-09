#!/bin/sh

# If inferring with the llama model, set 'use_lora' to 'False' and 'prompt_template' to 'ori_template'.
# If inferring with the default alpaca model, set 'use_lora' to 'True', 'lora_weights' to 'tloen/alpaca-lora-7b', and 'prompt_template' to 'alpaca'.
# If inferring with the llama-med model, download the LORA weights and set 'lora_weights' to './lora-llama-med' (or the exact directory of LORA weights) and 'prompt_template' to 'med_template'.

BASE_MODEL="decapoda-research/llama-7b-hf"
# 原始llama
# 这里在验证，所以不使用lora
o_cmd="python infer.py \
    --base_model ${BASE_MODEL} \
    --use_lora False \
    --prompt_template 'ori_template'"

# Alpaca
a_cmd="python infer.py \
    --base_model ${BASE_MODEL} \
    --use_lora True \
    --lora_weights "tloen/alpaca-lora-7b" \
    --prompt_template 'alpaca'"

# llama-med
m_cmd="python infer.py \
    --base_model ${BASE_MODEL} \
    --use_lora True \
    --lora_weights "lora-llama-med" \
    --prompt_template 'med_template'"

# 打印提示信息，表示正在运行原始llama模型
echo "ori"
# 执行原始llama模型的推理命令，并将输出重定向到o_tmp.txt文件
eval $o_cmd > infer_result/o_tmp.txt
# 打印提示信息，表示正在运行alpaca模型
echo "alpaca"
# 执行alpaca模型的推理命令，并将输出重定向到a_tmp.txt文件
eval $a_cmd > infer_result/a_tmp.txt
echo "med"
eval $m_cmd > infer_result/m_tmp.txt