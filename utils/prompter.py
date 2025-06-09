"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    # 限制实例属性，只允许template和_verbose
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose # 设置verbose，用于控制是否打印调试信息
        if not template_name:
            # 如果template_name为空，则使用默认模板
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json") # 拼接模板文件路径
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}") # 如果模板文件不存在，则抛出异常
        with open(file_name, encoding='utf-8') as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,  # 指令文本
        input: Union[None, str] = None,  # 可选的输入文本
        label: Union[None, str] = None,  # 可选的标签(响应/输出)文本
    ) -> str:
        # 根据指令和可选的输入生成完整的提示文本
        # 如果提供了标签(响应/输出)，也会被追加到提示文本中
        if input:
            # 如果有输入文本，使用带输入的模板格式
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            # 如果没有输入文本，使用不带输入的模板格式
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            # 如果有标签，将其追加到提示文本末尾
            res = f"{res}{label}"
        if self._verbose:
            # 如果启用了详细模式，打印生成的提示文本
            print(res)
        return res

    def get_response(self, output: str) -> str:
        # 从输出文本中提取响应部分
        # 使用模板中的response_split作为分隔符分割文本，取第二部分并去除首尾空白
        # 通常，split会是这样：“### 回答:”，用来分割输出文本
        return output.split(self.template["response_split"])[1].strip()
