import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# 这个文件用于创建一个基于Gradio的Web界面，用于模型推理和交互
# 它包含推理函数evaluate
try:
    # MPS是Apple的Metal框架的一部分，用于在Apple Silicon芯片上运行PyTorch模型
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "med_template",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.0.0.0'
    share_gradio: bool = True,
):
    # 确保指定了基础模型，否则会报错
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # 创建提示模板对象，用于生成提示文本，这里默认是med_template
    prompter = Prompter(prompt_template)
    # 加载分词器，用于将文本转换为token
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # 如果使用CUDA，则加载模型
    if device == "cuda":
        # 加载模型，使用8位量化加载模型以节省显存
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # 加载LoRA权重，将LoRA权重应用到模型中
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:#说明使用的是CPU
        # 使用低内存使用模式加载模型到当前设备
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    # 设置填充标记的ID为0，表示未知标记
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # 设置开始标记的ID为1
    model.config.bos_token_id = 1
    # 设置结束标记的ID为2
    model.config.eos_token_id = 2

    if not load_8bit:  # 只有在不使用8位量化时才执行
        model.half()  # 将模型参数从float32（32位浮点数）转换为float16（16位浮点数），可以减少模型占用的显存，加快推理速度

    model.eval()#设置模型为评估模式，禁用dropout等训练相关的操作
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,  # 设置top-p采样的概率阈值，表示累积概率达到p时，停止采样
        top_k=40,    # 设置top-k采样的候选词数量，只考虑概率最高的k个词
        num_beams=4,  # 设置束搜索的束宽，数值越大，生成质量越好，但速度越慢
        # 束搜索：每一步都保留多个可能的序列，最后选择累积概率最高的序列作为输出
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        # 使用tokenizer将提示文本转换为模型输入格式，return_tensors="pt"表示返回PyTorch张量
        inputs = tokenizer(prompt, return_tensors="pt")
        # 将输入张量移动到指定设备(CPU/GPU/MPS)上
        input_ids = inputs["input_ids"].to(device)
        # 创建生成配置对象，设置生成参数
        generation_config = GenerationConfig(
            temperature=temperature,  # 控制生成文本的随机性
            top_p=top_p,            # 控制核采样的概率阈值
            top_k=top_k,            # 控制每次只考虑概率最高的k个词
            num_beams=num_beams,    # 设置束搜索的束宽
            **kwargs,               # 其他生成参数
        )
        # 在推理时禁用梯度计算，节省内存
        with torch.no_grad():
            # 使用模型生成文本
            generation_output = model.generate(
                input_ids=input_ids,                    # 输入序列
                generation_config=generation_config,    # 生成配置
                return_dict_in_generate=True,          # 返回字典格式的结果
                output_scores=True,                    # 输出每个token的分数
                max_new_tokens=max_new_tokens,         # 最大生成token数
            )
        # 获取生成序列中的第一个序列
        s = generation_output.sequences[0]
        # 将生成的token序列解码为文本
        output = tokenizer.decode(s)
        # 从输出中提取实际响应内容
        return prompter.get_response(output)

    # 创建Gradio界面
    gr.Interface(
        fn=evaluate,  # 设置界面调用的函数为evaluate函数
        inputs=[
            # 创建指令输入框
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            # 创建输入框
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            # 创建温度参数滑动条
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            # 创建top-p参数滑动条
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            # 创建top-k参数滑动条
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            # 创建束搜索参数滑动条
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            # 创建最大token数滑动条
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[
            # 创建输出文本框
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="BenTsao",  # 设置界面标题
        description="",    # 设置界面描述
    ).launch(server_name=server_name, share=share_gradio)  # 启动Gradio服务器
    # Old testing code follows.

# 下面是一个例子，只输入Instruction，就能通过evaluate函数，获得输出
    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """


if __name__ == "__main__":
    fire.Fire(main)
