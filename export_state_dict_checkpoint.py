import json
import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: E402

BASE_MODEL = os.environ.get("BASE_MODEL", None)
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=decapoda-research/llama-7b-hf`"  # noqa: E501
# 这个文件用来将LoRA权重导出为state_dict格式，以便合并到基础模型

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

lora_model = PeftModel.from_pretrained(
    base_model,
    "tloen/alpaca-lora-7b",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

# merge weights
# 遍历模型的所有层
for layer in lora_model.base_model.model.model.layers:
    # 设置q查询投影层的权重合并标志为True
    layer.self_attn.q_proj.merge_weights = True
    # 设置v值投影层的权重合并标志为True
    layer.self_attn.v_proj.merge_weights = True

# 将模型设置为评估模式，禁用dropout（不正则化），禁用更新
lora_model.train(False)

# 获取模型的状态字典，通常包含各层权重
lora_model_sd = lora_model.state_dict()

# 定义模型参数配置
params = {
    "dim": 4096,        # 模型维度
    "multiple_of": 256, # 用于确保某些维度是256的倍数
    "n_heads": 32,      # 注意力头的数量
    "n_layers": 32,     # 模型层数
    "norm_eps": 1e-06,  # 层归一化的epsilon值，防止除以0
    "vocab_size": -1,   # 词汇表大小，-1表示未指定
}

# 从参数配置中提取关键参数
n_layers = params["n_layers"]  # 获取层数
n_heads = params["n_heads"]    # 获取注意力头数
dim = params["dim"]            # 获取模型维度
dims_per_head = dim // n_heads # 计算每个注意力头的维度

# 设置RoPE(旋转位置编码)的基础值
base = 10000.0

# 计算RoPE(旋转位置编码)的逆频率
# base是10000.0，作为RoPE的基础值
# torch.arange(0, dims_per_head, 2)生成一个从0到dims_per_head的偶数序列
# .float()将序列转换为浮点数
# dims_per_head是每个注意力头的维度
# 整个表达式计算了每个位置对应的逆频率值，用于生成旋转位置编码
inv_freq = 1.0 / (
    base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
)


# 定义permute函数，用于重排权重矩阵的维度
def permute(w):
    # 原始输入形状为(dim, dim)
    # 将输入张量w重塑为(n_heads, dim//n_heads//2, 2, dim)的形状
    # 然后转置第1和第2维
    # 最后重塑为(dim, dim)的形状
    return (
        w.view(n_heads, dim // n_heads // 2, 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )

# 定义unpermute函数，用于恢复权重矩阵的原始维度
def unpermute(w):
    # 将输入张量w重塑为(n_heads, 2, dim//n_heads//2, dim)的形状
    # 然后转置第1和第2维
    # 最后重塑为(dim, dim)的形状
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )

# 定义translate_state_dict_key函数，用于转换模型状态字典中的键名
def translate_state_dict_key(k):  # noqa: C901
    # 移除键名中的"base_model.model."前缀
    k = k.replace("base_model.model.", "")
    # 根据不同的键名模式返回对应的新键名
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        # 提取层号
        layer = k.split(".")[2]
        # 根据不同的层组件返回对应的新键名
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            # 如果遇到未知的键名模式，打印层号和键名并抛出异常
            print(layer, k)
            raise NotImplementedError
    else:
        # 如果遇到未知的键名模式，打印键名并抛出异常
        print(k)
        raise NotImplementedError


# 创建一个新的状态字典，用于存储转换后的模型参数
new_state_dict = {}
# 遍历原始LoRA模型的状态字典中的每个键值对
for k, v in lora_model_sd.items():
    # 使用translate_state_dict_key函数转换键名
    new_k = translate_state_dict_key(k)
    # 如果转换后的键名不为None
    if new_k is not None:
        # 如果键名中包含"wq"或"wk"，说明是查询或键的权重矩阵
        if "wq" in new_k or "wk" in new_k:
            # 使用unpermute函数重新排列权重矩阵的维度
            new_state_dict[new_k] = unpermute(v)
        else:
            # 对于其他类型的权重，直接保存
            new_state_dict[new_k] = v

# 创建保存检查点的目录，如果目录已存在则不报错
os.makedirs("./ckpt", exist_ok=True)

# 将转换后的状态字典保存为PyTorch模型文件
torch.save(new_state_dict, "./ckpt/consolidated.00.pth")

# 打开参数配置文件
with open("./ckpt/params.json", "w") as f:
    # 将模型参数配置保存为JSON格式，它包含模型维度、注意力头数、层数等
    json.dump(params, f)
