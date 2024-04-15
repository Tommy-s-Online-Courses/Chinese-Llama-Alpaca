# 开源项目 Chinese-LLaMA-AIpaca



## 第一部分：论文讲解

### 1. 论文部分内容解读；





## 第二部分：案例实战



### 1. 项目概览（文档 + 代码文件）



### 2. 案例实战（测试用例 + 环境配置）



### 第1步：生成指令数据

#### 脚本-1 : crawl_prompt.py

运行命令如下：

```shell
python scripts/crawl_prompt.py test/output.txt
```



### 第2步：将原版（官方）LLaMA模型转换为HF格式

<font color=blue>**LLaMA原版模型下载** </font>

```shell
链接: https://pan.baidu.com/s/17_y9BTpFys5SAhphL_EBEw 
提取码: nv8b
```

使用huggingface transformers提供的脚本 `convert_llama_weights_to_hf.py`，将原版LLaMA模型转换为HuggingFace格式。将原版LLaMA的 `tokenizer.model` 放在 `--input_dir` 指定的目录，其余文件放在`${input_dir}/${model_size}`下。执行以下命令后，`--output_dir`中将存放转换好的HF版权重。

- 可以通过 `git clone https://github.com/huggingface/transformers.git` 下载transformers

具体的命令如下（**<font color=red>注意：路径用你自己的路径，不要用我的路径</font>**）

① 进入到transformers相应的路径下，如下:

```shell
cd /Users/tgl/Downloads/transformers/src/transformers/models/llama
```

② 然后执行下面的命令

```shell
python convert_llama_weights_to_hf.py \
--input_dir /Users/tgl/Downloads/llama_models/ \
--model_size 7B \
--output_dir /Users/tgl/Downloads/llama_models/7B_hf/
```

如果一切正常，则会输出下面的结果:

**png-01**

![](./images/01.png)



### 第3步：合并LoRA权重，生成全量模型权重

#### 方式1：单LoRA权重合并（适用于 Chinese-LLaMA, Chinese-LLaMA-Plus, Chinese-Alpaca）

[链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2#%E5%8D%95lora%E6%9D%83%E9%87%8D%E5%90%88%E5%B9%B6%E9%80%82%E7%94%A8%E4%BA%8E-chinese-llama-chinese-llama-plus-chinese-alpaca)

#### 方式2：多LoRA权重合并（适用于Chinese-Alpaca-Plus ）

进入到项目路径下，执行下面的命令：

```shell
python scripts/merge_llama_with_chinese_lora.py \
--base_model /Users/tgl/Downloads/llama_models/7B_hf/ \
--lora_model /Users/tgl/Downloads/llama_models/chinese_llama_plus_lora_7b,/Users/tgl/Downloads/llama_models/chinese_alpaca_plus_lora_7b \
--output_dir /Users/tgl/Downloads/llama_models/7B_full_model
```

如果一切正常，则会输出下面的结果:

**png-02**

![](./images/02.png)



### 第4步：使用 Transformers 进行推理

在不安装其他库或Python包的情况下快速体验模型效果，可以使用 [scripts/inference_hf.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/scripts/inference_hf.py) 脚本启动非量化模型。该脚本支持CPU和GPU的单卡推理。以启动Chinese-Alpaca-7B模型为例，脚本运行方式，参考代码如下：

```python
CUDA_VISIBLE_DEVICES={device_id} python scripts/inference_hf.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_or_alpaca_lora \
    --with_prompt \
    --interactive
```

如果已经执行了`merge_llama_with_chinese_lora_to_hf.py`脚本将lora权重合并，那么无需再指定`--lora_model`，启动方式更简单，实际执行代码如下：

```shell
python scripts/inference_hf.py \
--base_model /Users/tgl/Downloads/llama_models/7B_full_model \
--with_prompt \
--interactive
```

参数说明：

- `{device_id}`：CUDA设备编号。如果为空，那么在CPU上进行推理
- `--base_model {base_model} `：存放**HF格式**的LLaMA模型权重和配置文件的目录。如果之前合并生成的是PyTorch格式模型，[请转换为HF格式](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/使用Transformers推理#step-1-将原版llama模型转换为hf格式)
- `--lora_model {lora_model}` ：中文LLaMA/Alpaca LoRA解压后文件所在目录，也可使用[🤗Model Hub模型调用名称](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/使用Transformers推理#Model-Hub)。若不提供此参数，则只加载`--base_model`指定的模型
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与`--lora_model`相同；若也未提供`--lora_model`参数，则其默认值与`--base_model`相同
- `--with_prompt`：是否将输入与prompt模版进行合并。**如果加载Alpaca模型，请务必启用此选项！**
- `--interactive`：以交互方式启动，以便进行多次**单轮问答**（此处不是llama.cpp中的上下文对话）
- `--data_file {file_name}`：非交互方式启动下，按行读取`file_name`中的的内容进行预测
- `--predictions_file {file_name}`：非交互式方式下，将预测的结果以json格式写入`file_name`

注意事项：

- 因不同框架的解码实现细节有差异，该脚本并不能保证复现llama.cpp的解码效果
- 该脚本仅为方便快速体验用，并未对多机多卡、低内存、低显存等情况等条件做任何优化
- 如在CPU上运行7B模型推理，请确保有32GB内存；如在GPU上运行7B模型推理，请确保有20GB显存



### 第5步：使用 webui 搭建界面

接下来以[text-generation-webui工具](https://github.com/oobabooga/text-generation-webui)为例，介绍无需合并模型即可进行**本地化部署**的详细步:

##### Step 1: 克隆text-generation-webui并安装必要的依赖

```shell
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```



##### Step 2: 将下载后的lora权重放到loras文件夹下

```shell
ls models/llama-7b-hf

pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin config.json pytorch_model.bin.index.json generation_config.json
```



##### Step 3: 将HuggingFace格式的llama-7B模型文件放到models文件夹下



##### Step 4: 复制lora权重的tokenizer到models/llama-7b-hf下(webui默认从`./models`下加载tokenizer.model,因此需使用扩展中文词表后的tokenizer.model)

```shell
cp loras/chinese-alpaca-lora-7b/tokenizer.model models/llama-7b-hf/
cp loras/chinese-alpaca-lora-7b/special_tokens_map.json models/llama-7b-hf/
cp loras/chinese-alpaca-lora-7b/tokenizer_config.json models/llama-7b-hf/
```



##### Step 5: 修改/modules/LoRA.py文件，在`PeftModel.from_pretrained`方法之前添加一行代码修改原始llama的embed_size

```shell
shared.model.resize_token_embeddings(len(shared.tokenizer))
shared.model = PeftModel.from_pretrained(shared.model, Path(f"{shared.args.lora_dir}/{lora_name}"), **params)
```



##### Step 6: 接下来就可以愉快的运行了，参考[webui using LoRAs](https://github.com/oobabooga/text-generation-webui/wiki/Using-LoRAs)，您也可以直接运行合并后的chinese-alpaca-7b，相对加载两个权重推理速度会有较大的提升



### 第6步：基于LLaMA模型和LoRA模型进行预训练

省略，直接看B站分享的视频。





### 第7步：案例实战：指令精调脚本

#### 7.1 单卡训练

##### 7.1.1 情景1：继续训练 Chinese-AIpaca模型的LoRA权重

- `--model_name_or_path`: 原版HF格式LLaMA模型（如果继续训练非Plus Alpaca模型）**或**合并Chinese-LLaMA-Plus-LoRA后的Chinese-LLaMA模型（如果继续训练Plus模型）

- `--peft_path`: Chinese-Alpaca的LoRA权重目录

- 无需指定`--lora_rank`、`--lora_alpha`、`--lora_dropout`、`--trainable`和`--modules_to_save`参数

  

① 合并Chinese-LLaMA-Plus-LoRA后的Chinese-LLaMA模型（如果继续训练Plus模型）

`run_sft.sh的内容如下`

```shell
########参数部分########
lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05
pretrained_model=/root/autodl-tmp/7B_full_model
chinese_tokenizer_path=/root/autodl-tmp/projects/scripts/merged_tokenizer_hf
dataset_dir=/root/autodl-tmp/projects/data/train
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=10
gradient_accumulation_steps=1
output_dir=/root/autodl-tmp/sft_output
peft_model=/root/autodl-tmp/chinese_alpaca_plus_lora_7b
validation_file=/root/autodl-tmp/projects/data/eval/Belle_open_source_0.5M.json

deepspeed_config_file=ds_zero2_no_offload.json

########启动命令########
torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --peft_path ${peft_model} \
    --ddp_find_unused_parameters False
```





##### 7.1.2 情景2：基于中文Chinese-LLaMA训练全新的指令精调LoRA权重

**第1步：合并HF格式的LLaMA模型与chinese_llama_plus_lora_7b模型**

命令如下（注意路径）：

```shell
python scripts/merge_llama_with_chinese_lora.py \
--base_model /root/autodl-tmp/llama_7b_hf/ \
--lora_model /root/autodl-tmp/chinese_llama_plus_lora_7b/ \
--output_type huggingface \
--output_dir /root/autodl-tmp/merge_llama_with_llama_lora_hf/
```

**第2步：合并后检查（重要！）**

Chinese-LLaMA-Plus-7B 的 SHA256 :

`f8d380d63f77a08b7f447f5ec63f0bb1cde9ddeae2207e9f86e6b5f0f95a7955`

命令如下：

```shell
sha256sum consolidated.00.pth
```

截图如下：

![](./images/03.png)

**第3步：基于中文Chinese-LLaMA训练全新的指令精调LoRA权重**

- `--model_name_or_path`: 合并对应Chinese-LLaMA-LoRA后的HF格式Chinese-LLaMA模型（无论是否是Plus模型）
- `--peft_path`: 勿提供此参数，并且从脚本中删除 `--peft_path`
- 需指定`--lora_rank`、`--lora_alpha`、`--lora_dropout`、`--trainable`和`--modules_to_save`参数

```shell
########参数部分########
lr=1e-4
lora_rank=8
lora_alpha=32
#lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_trainable="q_proj,v_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/root/autodl-tmp/merge_llama_with_llama_lora_hf
chinese_tokenizer_path=/root/autodl-tmp/projects/scripts/merged_tokenizer_hf
dataset_dir=/root/autodl-tmp/projects/data/train
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=100
gradient_accumulation_steps=1
output_dir=/root/autodl-tmp/sft_output
validation_file=/root/autodl-tmp/projects/data/eval/Belle_open_source_0.5M.json

deepspeed_config_file=ds_zero2_no_offload.json

########启动命令########
torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
```



### 第8步：与LangChain进行集成

[参考官方链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E4%B8%8ELangChain%E8%BF%9B%E8%A1%8C%E9%9B%86%E6%88%90)

#### 8.1 如何在LangChain中使用Chinese-Alpaca？

因为将LoRA权重合并进LLaMA后的模型与原版LLaMA除了词表不同之外结构上没有其他区别，因此可以参考任何基于LLaMA的LangChain教程进行集成。 以下文档通过两个示例，分别介绍在LangChain中如何使用Chinese-Alpaca实现

- 检索式问答
- 摘要生成

例子中的超参、prompt模版均未调优，仅供演示参考用。



##### 8.1.1 准备工作

**第1步：环境准备**

```shell
pip install langchain
pip install sentence_transformers faiss-cpu
```

**从源码安装[commit id为13e53fc的Peft](https://github.com/huggingface/peft/tree/13e53fc)**

```shell
pip install git+https://github.com/huggingface/peft.git@13e53fc
```



**第2步：模型合并**

* 多LoRA权重合并（适用于Chinese-Alpaca-Plus ）

```shell
python scripts/merge_llama_with_chinese_lora.py \
--base_model /root/autodl-tmp/7B_hf/ \
--lora_model /root/autodl-tmp/chinese_llama_plus_lora_7b/,/root/autodl-tmp/chinese_alpaca_plus_lora_7b \
--output_type huggingface \ 
--output_dir /root/autodl-tmp/7B_merge_model/
```

⚠️ **两个LoRA模型的顺序很重要，不能颠倒。先写LLaMA-Plus-LoRA然后写Alpaca-Plus-LoRA。**



**第3步：合并后检查（重要！）**

<font color=red>**合并完成后务必检查SHA256！这里是针对于pth格式的模型，而非huggingface格式的模型**</font>

```shell
sha256sum consolidated.00.pth
```



##### 8.1.2 检索式问答

该任务使用LLM完成针对特定文档的自动问答，流程包括：文本读取、文本分割、文本/问题向量化、文本-问题匹配、将匹配文本作为上下文和问题组合生成对应Prompt中作为LLM的输入、生成回答。

```shell
python langchain_qa.py \
--embedding_path /root/autodl-tmp/text2vec-large-chinese/ \
--model_path /root/autodl-tmp/7B_merge_model/ \
--file_path /root/autodl-tmp/Chinese-LLaMA-AIpaca-4.0/scripts/langchain/doc.txt \
--chain_type refine
```

参数说明：

- `--embedding_path`: 下载至本地的embedding model所在目录（如`text2vec-large-chinese`）或HuggingFace模型名（如`GanymedeNil/text2vec-large-chinese`）
- `--model_path`: 合并后的Alpaca模型所在目录
- `--file_path`: 待进行检索与提问的文档
- `--chain_type`: 可以为`refine`(默认)或`stuff`，为两种不同的chain，详细解释见[这里](https://docs.langchain.com/docs/components/chains/index_related_chains)。简单来说，stuff适用于较短的篇章，而refine适用于较长的篇章。
- `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如`0,1,2`

⚠️ **修改LangChain中`langchain.HuggingFacePipeline.from_model_id`中的相关代码，将其中tokenizer的初始化部分**

```shell
代码路径：/root/miniconda3/lib/python3.8/site-packages/langchain/llms
```

```shell
tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
```

修改为：

```shell
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **_model_kwargs)
```



##### 8.1.2 摘要生成

该任务使用LLM完成给定文档的摘要生成，以帮助提炼文档中的核心信息。

```shell
python langchain_sum.py \
--model_path /root/autodl-tmp/7B_merge_model/ \
--file_path /root/autodl-tmp/Chinese-LLaMA-AIpaca-4.0/scripts/langchain/sample.txt \
--chain_type refine
```

参数说明：

- `--model_path`: 合并后的Alpaca模型所在目录
- `--file_path`: 待进行摘要的文档
- `--chain_type`: 可以为`refine`(默认)或`stuff`，为两种不同的chain，详细解释见[这里](https://docs.langchain.com/docs/components/chains/index_related_chains)。简单来说，stuff适用于较短的篇章，而refine适用于较长的篇章。
- `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如`0,1,2`





### 第9步：使用FastAPI添加OpenAI API 示例

#### 9.1 环境配置

新增安装相关的依赖包：

```shell
pip install requests fastapi uvicorn shortuuid
```



#### 9.2 启动服务

```shell
python openai_api_server.py --base_model /root/autodl-tmp/7B_merge_model/ --tokenizer_path /root/autodl-tmp/7B_merge_model/tokenizer.model
```



#### 9.3 测试 APIs

① 测试 completions

```shell
curl http://localhost:19327/v1/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "prompt": "告诉我中国的首都在哪里"
  }'
```



② 测试 chat completions

```shell
curl http://localhost:19327/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "messages": [
      {"role": "user","message": "给我讲一些有关杭州的故事吧"}
    ],
    "repetition_penalty": 1.0
  }'
```



③ 测试 embeddings

```shell
curl http://localhost:19327/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "今天天气真不错"
  }'
```













## 第三部分：源码讲解（核心代码逐一讲解）



### 代码-1 crawl_prompt.py &#x2705;



### 代码-2 merge_tokenizers.py &#x2705;



### 代码-3 merge_llama_with_lora.py &#x2705;



### 代码-4 inference_hf.py &#x2705;



