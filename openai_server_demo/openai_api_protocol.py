from typing import Optional, List, Dict, Any, Union

import time

import shortuuid
# Pydantic是一个Python数据解析和验证库，它可以自动转换和验证数据类型。
from pydantic import BaseModel, Field

class ChatCompletionRequest(BaseModel):
    model: str = "chinese-llama-alpaca"
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7 # 用于控制生成文本的随机性的参数
    top_p: Optional[float] = 1.0 #  用于控制生成文本的多样性的参数，值越大，生成的文本越多样；值越小，生成的文本越聚焦。
    top_k: Optional[int] = 40 # 在文本生成时，每步只保留最可能的k个结果
    n: Optional[int] = 1 # 要生成的回复的数量
    max_tokens: Optional[int] = 128 # 生成的文本的最大长度
    num_beams: Optional[int] = 4 # Beam Search策略
    stop: Optional[Union[str, List[str]]] = None # 生成文本时的停止标记，当遇到这些标记时，停止生成文本。
    stream: Optional[bool] = False # 是否启用流模式，流模式会在生成每个token后立即返回，不等待整个序列生成完。
    repetition_penalty: Optional[float] = 1.0 # 重复惩罚因子，用于控制生成的文本中的重复情况，默认值为1.0，表示不设惩罚。
    user: Optional[str] = None # 用户标识


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    #
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    # default_factory参数用于提供一个工厂函数，当没有为created提供值的时候，会调用这个函数生成默认值。
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "chinese-llama-alpaca"
    choices: List[ChatCompletionResponseChoice]


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[Any]]
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str = "chinese-llama-alpaca"


class CompletionRequest(BaseModel):
    prompt: Union[str, List[Any]]
    temperature: Optional[float] = 0.1
    n: Optional[int] = 1
    max_tokens: Optional[int] = 128
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 0.75
    top_k: Optional[int] = 40
    num_beams: Optional[int] = 4
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0
    user: Optional[str] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str


class CompletionResponse(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: Optional[str] = "text_completion"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = 'chinese-llama-alpaca'
    choices: List[CompletionResponseChoice]
