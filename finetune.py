import os
import sys
from typing import List

import fire
import wandb
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# 写了一个train函数
# 包含：配置分布式训练参数；初始化模型和分词器；加载数据；配置训练参数；训练模型；保存模型
def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 500,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "llama_med",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    # 总批次大小除以微批次大小来计算梯度累积步数
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)  # 例如alpaca。用于生成训练数据的提示模板

    device_map = "auto"# 自动分配设备
    world_size = int(os.environ.get("WORLD_SIZE", 1))# 获取分布式训练的进程数
    ddp = world_size != 1# 判断是否使用分布式训练
    if ddp:# 如果使用分布式训练
        # 在分布式训练环境下，将模型分配到对应的GPU设备上
        # LOCAL_RANK环境变量表示当前进程的本地GPU编号，如果未设置则默认为0
        # 这样设置可以确保在分布式训练时，每个进程的模型都被正确地分配到对应的GPU上。每个进程有不同的Local_rank
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        
        # 在分布式训练中，由于数据被分配到多个GPU上并行处理
        # 需要将梯度累积步数除以总进程数，以保持每个GPU上的实际批次大小一致
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # 从预训练模型加载
    model = AutoModelForCausalLM.from_pretrained(
        base_model,  # 基础模型路径
        load_in_8bit=True,  # 使用8位量化加载模型以节省显存
        torch_dtype=torch.float16,  # 使用半精度浮点数
        device_map=device_map,  # 指定模型加载的设备映射
    )

    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 设置填充标记的ID为0，后续遇到0就不处理。要注意，他不能等于eos_token_id（结束标记）
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    # 设置填充位置在左侧，以支持批处理推理
    # 因为：一般是从左到右生成文本的，如果填充在右侧，会导致模型在计算时看到很多无意义的填充token，这些填充token会影响模型的注意力计算
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # 对输入文本进行分词处理，获得input_ids和attention_mask
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,  # 输入文本
            truncation=True,  # 启用截断
            max_length=cutoff_len,  # 最大长度限制
            padding=False,  # 不进行填充，因为这是对单个样本的处理，不是批处理，还不知道要填充成多长
            return_tensors=None,  # 返回列表而不是张量
        )
        # 检查是否需要添加结束标记
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id  # 如果最后一个token不是结束标记
            and len(result["input_ids"]) < cutoff_len  # 且长度未超过限制
            and add_eos_token  # 且需要添加结束标记
        ):
            # 添加结束标记到input_ids
            result["input_ids"].append(tokenizer.eos_token_id)
            # 添加对应的attention mask为1，表示这个token是有效的
            result["attention_mask"].append(1)

        # 复制input_ids作为标签
        result["labels"] = result["input_ids"].copy()

        return result

    # 定义生成和标记提示的函数
    def generate_and_tokenize_prompt(data_point):
        # 生成完整的提示文本，包含指令、输入和输出
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        # 对完整提示进行分词处理。默认需要添加eos_token_id
        tokenized_full_prompt = tokenize(full_prompt)
        # 如果不训练输入部分
        if not train_on_inputs:
            # 根据指令和输入，生成用户提示
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            # 对用户提示进行分词，不添加结束标记
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            # 获取用户提示的长度
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # 将用户提示部分的标签（也就是前user_prompt_len个token）设为-100（表示不计算损失）
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # 可以优化速度
        return tokenized_full_prompt

    # 准备模型进行int8训练（int8量化）
    model = prepare_model_for_int8_training(model)

    # 配置LoRA参数
    config = LoraConfig(
        r=lora_r,  # LoRA的秩，经验值是8或16，值越小，训练更快，显存占用更少
        # LoRA的缩放因子，与r配合使用，实际更新量 = 更新值 * (alpha/r)，较大的alpha值会使模型更新更显著。通常设置为r的2倍
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,  # 需要应用LoRA的目标模块。通常只对注意力机制中的查询和值投影矩阵进行微调
        lora_dropout=lora_dropout,  # LoRA的dropout率，较大的值会增加正则化强度，防止过拟合。默认0.05
        bias="none",  # 不训练偏置项，因为偏置项对模型性能影响较小。这样可以进一步减少参数量
        task_type="CAUSAL_LM",  # 任务类型为因果语言模型
    )
    # 将模型转换为PEFT模型，PEFT是参数高效微调，可以只训练部分参数，而不是整个模型
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"#这里的bin表示模型权重文件
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - 加载的LoRA参数必须与当前配置匹配
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # 打印可训练参数的数量和百分比
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # 如果验证集大小大于0，则进行训练集和验证集的划分
    if val_set_size > 0:
        # 将数据集按照val_set_size的比例划分为训练集和验证集，设置随机种子为2023
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=2023
        )
        # 对训练集进行打乱并应用generate_and_tokenize_prompt函数处理
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        # 对验证集进行打乱并应用generate_and_tokenize_prompt函数处理
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        # 如果没有验证集，只处理训练集。取出训练集，打乱，应用generate_and_tokenize_prompt函数，获得tokenized_full_prompt
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # 如果不是分布式训练且GPU数量大于1，说明单机多卡，设置模型并行化参数
    if not ddp and torch.cuda.device_count() > 1:
        # 防止Trainer在多个GPU可用时尝试自己的DataParallelism（数据并行），这样的话，每个GPU必须能放下完整模型
        # 允许模型并行化，将模型的不同部分放在不同GPU上，每个GPU只负责模型的一部分
        model.is_parallelizable = True
        model.model_parallel = True

    # 定义保存PEFT模型的回调类
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            # 构建检查点文件夹路径
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            # 保存模型到检查点文件夹
            kwargs["model"].save_pretrained(checkpoint_folder)
            # 构建pytorch模型文件路径
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            # 如果已经存在pytorch模型文件，则删除
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control# 返回控制权，让Trainer继续执行

    # 创建训练器实例，用于模型训练
    trainer = transformers.Trainer(
        # 设置要训练的模型
        model=model,
        # 设置训练数据集
        train_dataset=train_data,
        # 设置验证数据集
        eval_dataset=val_data,
        # 设置训练参数
        args=transformers.TrainingArguments(
            # 每个设备的训练批次大小，代表每个GPU一次处理的样本数
            per_device_train_batch_size=micro_batch_size,
            # 梯度累积步数
            gradient_accumulation_steps=gradient_accumulation_steps,
            # 预热比例，表示在训练开始时，学习率从0逐渐增加到最大值的10%
            warmup_ratio=0.1,
            # 训练轮数
            num_train_epochs=num_epochs,
            # 学习率
            learning_rate=learning_rate,
            # 启用半精度训练
            fp16=True,
            # 日志记录步数
            logging_steps=8,
            # 使用AdamW优化器
            optim="adamw_torch",
            # 评估策略：如果有验证集则按步数评估，否则不评估
            # 按步数评估，每32步评估一次
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            # 保存策略：按步数保存
            save_strategy="steps",
            # 评估步数：如果有验证集则每32步评估一次
            eval_steps=32 if val_set_size > 0 else None,
            # 保存步数：每32步保存一次
            save_steps=32,
            # 输出目录
            output_dir=output_dir,
            # 最多保存5个检查点
            save_total_limit=5,
            # 是否在训练结束时加载最佳模型
            load_best_model_at_end=True if val_set_size > 0 else False,
            # 分布式训练参数设置
            ddp_find_unused_parameters=False if ddp else None,
            # 是否按token长度分组，如果为True，则将长度相近的样本放在同一个batch中，以提高GPU利用率，减少填充token的数量
            group_by_length=group_by_length,
            # 是否使用wandb进行实验跟踪
            report_to="wandb" if use_wandb else None,
            # wandb运行名称
            run_name=wandb_run_name if use_wandb else None,
        ),
        # 设置数据整理器，用于处理序列到序列任务的数据
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # 设置回调函数，用于保存LoRA权重
        callbacks=[SavePeftModelCallback],
    )
    # 禁用模型缓存，因为LoRA模型在推理时需要重新计算注意力权重，所以不需要缓存
    # 一般推理时，模型会缓存注意力权重，以提高推理速度
    model.config.use_cache = False

    # 如果PyTorch版本大于等于2.0且不是Windows系统，则使用torch.compile编译模型
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # 开始训练，如果指定了检查点则从检查点恢复训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
