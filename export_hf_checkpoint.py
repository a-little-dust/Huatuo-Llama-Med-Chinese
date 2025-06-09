import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

BASE_MODEL = os.environ.get("BASE_MODEL", None)
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=decapoda-research/llama-7b-hf`"  # noqa: E501
# 这个文件用来将LoRA权重导出为Hugging Face格式，以便合并到基础模型

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)
# 获取基础模型的第一层注意力层的q权重，这里只是为了检查权重能否合并
first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

# 加载LoRA权重
lora_model = PeftModel.from_pretrained(
    base_model,
    "tloen/alpaca-lora-7b",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

# 获取LoRA模型的第一层注意力层的q权重
lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

# 确保基础模型的第一层注意力层的q权重和原始模型的第一层注意力层的q权重相同
assert torch.allclose(first_weight_old, first_weight)

# 合并权重
# 遍历lora_model的每一层，将q权重和v权重设置为True，表示合并权重
for layer in lora_model.base_model.model.model.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

# 将模型设置为评估模式
lora_model.train(False)

# 确保权重合并成功，因为权重已经合并，所以权重应该不同
assert not torch.allclose(first_weight_old, first_weight)

# 获取LoRA模型的状态字典
lora_model_sd = lora_model.state_dict()
# 删除LoRA权重
deloreanized_sd = {
    k.replace("base_model.model.", ""): v # 删除键名中的"base_model.model."前缀
    for k, v in lora_model_sd.items()
    if "lora" not in k # 删除键名中包含"lora"的权重。这是因为前面已经合并到基础模型了
}

# 保存处理后的基础模型和deloreanized_sd
# max_shard_size=400MB，表示将模型分成多个shard文件，每个shard文件不超过400MB
LlamaForCausalLM.save_pretrained(
    base_model, "./hf_ckpt", state_dict=deloreanized_sd, max_shard_size="400MB"
)
