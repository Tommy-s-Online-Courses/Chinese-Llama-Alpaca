import sys
import argparse
import os
from fastapi import FastAPI
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--load_in_8bit', action='store_true', help='use 8 bit model')
parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
args = parser.parse_args()
# 是否量化
load_in_8bit = args.load_in_8bit
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel

from openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    EmbeddingsRequest,
    EmbeddingsResponse,
)

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
)
load_type = torch.float16 # 数据类型
# 部署到 device
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')
# 参数判断
if args.tokenizer_path is None:
    args.tokenizer_path = args.lora_model
    if args.lora_model is None:
        args.tokenizer_path = args.base_model
# 加载 tokenizer 模型
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
# 加载 预训练模型
base_model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    load_in_8bit=load_in_8bit,
    torch_dtype=load_type,
    low_cpu_mem_usage=True,
    device_map='auto',
)
# model词汇表大小
model_vocab_size = base_model.get_input_embeddings().weight.size(0)
# tokenizer词汇表大小
tokenzier_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
# 确保两者词汇表大小
if model_vocab_size != tokenzier_vocab_size:
    assert tokenzier_vocab_size > model_vocab_size
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenzier_vocab_size)
# 如果LoRA不为空，直接用peft加载
if args.lora_model is not None:
    print("loading peft model")
    model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto', )
else:
    model = base_model

if device == torch.device('cpu'):
    model.float()

model.eval() # 评估状态


def generate_completion_prompt(instruction: str):
    """Generate prompt for completion 指令模板"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {instruction}
            ### Response: """


def generate_chat_prompt(messages: list):
    """Generate prompt for chat completion 指令模板"""
    system_msg = '''A chat between a curious user and an artificial intelligence assistant. 
    The assistant gives helpful, detailed, and polite answers to the user's questions.'''
    # 情况1：role = 'system'
    for msg in messages:
        if msg.role == 'system':
            system_msg = msg.message
    prompt = f"{system_msg} <\s>"

    # 情况2：role != 'system'
    for msg in messages:
        if msg.role == 'system':
            continue
        prompt += f"{msg.role}: {msg.content} <\s>"
    prompt += "assistant:"

    return prompt


def predict(
        input,
        max_new_tokens=128,
        top_p=0.75,
        temperature=0.1,
        top_k=40,
        num_beams=4,
        repetition_penalty=1.0,
        **kwargs,
):
    """
    Main inference method
    type(input) == str -> /v1/completions
    type(input) == list -> /v1/chat/completions
    """
    if type(input) == str:
        prompt = generate_completion_prompt(input)
    else:
        prompt = generate_chat_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt") # 对 prompt 进行分词
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature, # 调节下一个token的输出概率
        top_p=top_p, # 默认p=1.0，用于控制生成文本的多样性的参数，值越大，生成的文本越多样
        top_k=top_k, # 默认k=50，k个最高概率词汇token的数量
        num_beams=num_beams, # beam search中beam的数量
        **kwargs,
    )
    # 生成
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(repetition_penalty), # 重复惩罚因子，用于控制生成的文本中的重复情况
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True) # 解码部分
    if type(input) == str:
        output = output.split("### Response:")[-1].strip()
    else:
        output = output.split("assistant:")[-1].strip()
    return output


def get_embedding(input):
    """先将输入的文本序列进行编码，然后用模型得到每个词的嵌入表示，
       对嵌入表示进行遮掩和求和，然后计算平均嵌入，最后进行归一化处理，
       得到每个序列的嵌入表示。"""
    with torch.no_grad():
        # 在输入序列长度不一致时，需要用填充符号对序列进行填充，使它们长度一致。
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # 使用tokenizer对输入的文本序列进行批量编码，
        # padding=True表示需要对序列进行填充，使其长度一致；
        # return_tensors="pt"表示返回的编码结果为Pytorch的tensor。
        encoding = tokenizer.batch_encode_plus(
            input, padding=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to('cuda')
        attention_mask = encoding["attention_mask"].to('cuda')
        # 使用模型对输入的input ids和attention mask进行处理，得到模型的输出
        # output_hidden_states=True表示需要返回所有层的隐藏状态。
        model_output = model(
            input_ids, attention_mask, output_hidden_states=True
        )
        # 从模型输出中获取最后一层的隐藏状态作为词的嵌入表示
        data = model_output.hidden_states[-1]
        # 对注意力掩码进行处理，使其维度与词嵌入的维度一致
        mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
        # 通过点乘操作，用处理后的mask对embedding进行遮掩，屏蔽掉填充词的嵌入。
        masked_embeddings = data * mask
        # 获取每个序列的embedding总和
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        # 计算每个序列的有效长度
        seq_length = torch.sum(mask, dim=1)
        # 对embedding总和进行平均
        embedding = sum_embeddings / seq_length
        # 对平均embedding进行L2归一化处理，这样处理后，可以减少embedding的量纲影响，使得embedding的距离度量更加准确。
        normalized_embeddings = F.normalize(embedding, p=2, dim=1)
        # 列表形式
        ret = normalized_embeddings.tolist()
    return ret


app = FastAPI()


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """接收用户的POST请求，获取并处理请求中的消息，调用预测函数得到预测结果，
       然后将预测结果和原始消息都封装为响应对象，最后返回该响应对象。"""
    msgs = request.messages
    if type(msgs) == str:
        msgs = [ChatMessage(role='user', content=msgs)]
    else:
        msgs = [ChatMessage(role=x['role'], content=x['message']) for x in msgs]
    # 调用预测函数
    output = predict(
        input=msgs,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
    )
    choices = [ChatCompletionResponseChoice(index=i, message=msg)
               for i, msg in enumerate(msgs)]
    choices += [ChatCompletionResponseChoice(index=len(choices),
                                             message=ChatMessage(role='assistant', content=output))]
    return ChatCompletionResponse(choices=choices)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """接收用户的POST请求，获取并处理请求中的提示语，调用预测函数得到预测结果，
       然后将预测结果封装为响应对象，最后返回该响应对象。"""
    output = predict(
        input=request.prompt,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
    )
    choices = [CompletionResponseChoice(index=0, text=output)]
    return CompletionResponse(choices=choices)


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """Creates text embedding"""
    embeddings = get_embedding(request.input)
    data = [{
        "object": "embedding",
        "embedding": embeddings[0],
        "index": 0
    }]
    return EmbeddingsResponse(data=data)


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG # Uvicorn的默认日志配置
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    # 运行Uvicorn服务器，参数的含义：
    # app是FastAPI的应用实例
    # host = '0.0.0.0' 表示服务器监听所有可用的网络接口
    # port = 19327 表示服务器监听的端口号是19327
    # workers = 1 表示服务器使用1个工作进程
    # log_config = log_config表示使用上面修改后的日志配置
    uvicorn.run(app,
                host='0.0.0.0',
                port=19327,
                workers=1,
                log_config=log_config)