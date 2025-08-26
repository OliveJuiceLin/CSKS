import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,trainer,training_args,BitsAndBytesConfig
import torch
from torch import nn
import einops
import os
from peft import LoraConfig,get_peft_model
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#量化选项
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,#在4bit上，进行量化
    bnb_4bit_use_double_quant=True,# 嵌套量化，每个参数可以多节省0.4位
    bnb_4bit_quant_type="nf4",#NF4（normalized float）或纯FP4量化 博客说推荐NF4
    bnb_4bit_compute_dtype=torch.float16,
)
model_id = ''
# 数据集
train_data_path = 'data/target_dataset_train_SHORT_SHORT.csv'
val_data_path = 'data/target_dataset_validation_SHORT_SHORT.csv'
df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)


# 数据集处理
def convert_to_instructive_format_1(example):
    question = example['question']
    context = example['context']
    choices = example['choices']
    
    # 格式化选项
    formatted_choices = " ".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    messages = [
    {"role": "system",
     "content": "You are a helpful AI assistant designed to answer questions. You should understand the context and content of every question and then choose the correct option to answer the question. Always answer in the format: 'Answer: [option]\n\nExplaination: [your explanation]'"},
    {"role": "user", 
     "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{formatted_choices}\n"}
    ]
    ans = example['c_answer']
    explanation = example['explanation']
    response = [{"role":"assistant", "content": f"Answer: {ans}\n\nExplanation: {explanation}"}]
    return {'input_text': messages, 'output_text': response}
def convert_to_instructive_format_2(example):
    question = example['question']
    context = example['context']
    choices = example['choices']
    
    # 格式化选项
    formatted_choices = " ".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    
    ans = example['c_answer']
    explanation = example['explanation']
    message = [
    {"role": "system",
     "content": "You are a helpful AI assistant designed to answer questions. You should understand the context and content of every question and then choose the correct option to answer the question. Always answer in the format: 'Answer: [option]\n\nExplaination: [your explanation]'"},
    {"role": "user", 
     "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{formatted_choices}\n"},
    {"role": "assistant", "content": f"Answer: {ans}\n\nExplaination: {explanation}"}
    ]
    return {'conversation': message}

choice = 'after'
if choice == 'after':
    convert_to_instructive_format = convert_to_instructive_format_2
elif choice == 'before':
    convert_to_instructive_format = convert_to_instructive_format_1

processed_dataset = dataset_train.map(convert_to_instructive_format, remove_columns=['question', 'context', 'choices', 'c_answer', 'explanation'])
processed_dataset_val = dataset_val.map(convert_to_instructive_format, remove_columns=['question', 'context', 'choices', 'c_answer', 'explanation'])


# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
#tokenizer.pad_token = tokenizer.eos_token### 修改
#量化?
base_model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config)
#base_model = AutoModelForCausalLM.from_pretrained(model_id,)
# 训练的时候右侧填充
tokenizer.padding_side = 'right'
# 生成的时候左侧填充
#tokenizer.padding_side = 'left'
# Qwen模型不需要
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

base_model.config.pretraining_tp = 1
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "down_proj"],
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

######################################################
# frozen等操作
for i, param in enumerate(base_model.parameters()):
    param.requires_grad = False  # freeze the model - train adapters later
#     print(i, 'param.requires_grad = False')
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)
#         print(i, 'ndim == 1, torch.float16 to torch.float32')
# reduce number of stored activations
base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()


#class CastOutputToFloat(nn.Sequential):
#    def forward(self, x):
#        return super().forward(x).to(torch.float32)
#base_model.lm_head = CastOutputToFloat(base_model.lm_head)

######################################################
model = get_peft_model(base_model, peft_config)
# Qwen模型不需要这一步
# with torch.no_grad():
#     model.resize_token_embeddings(len(tokenizer))
#     model.config.pad_token_id = tokenizer.pad_token_id  


# 进行tokenize
def tokenize_function_1(examples):
    # inputs
    template_inputs = tokenizer.apply_chat_template(examples['input_text'],tokenize=False,add_generation_prompt=True)
    inputs = tokenizer(template_inputs, padding="max_length", truncation=True, max_length=512)
    # outputs
    template_outputs = tokenizer.apply_chat_template(examples['output_text'],tokenize=False,add_generation_prompt=False)
    start_index = template_outputs.find('Answer:')
    template_outputs = template_outputs[start_index:]
    labels = tokenizer(template_outputs, padding="max_length", truncation=True, max_length=128)
    # 对 input_text 进行编码
    #inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)
    # 对 output_text 进行编码，并将其作为标签
    #labels = tokenizer(examples['output_text'], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs
def tokenize_function_2(examples):
    output_texts = []
    for c in examples["conversation"]:
        text = tokenizer.apply_chat_template(c, tokenize=False,add_generation_prompt=False)
        output_texts.append(text)
    return output_texts
if choice == 'after':
    tokenize_function = tokenize_function_2
elif choice == 'before':
    tokenize_function = tokenize_function_1

if choice == 'before':
    tokenized_dataset = processed_dataset.map(tokenize_function,batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_dataset_val = processed_dataset_val.map(tokenize_function, batched=True)
    tokenized_dataset_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    output_dir = ""
    training_args = training_args.TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,            # 学习率
        per_device_train_batch_size=4,  # 训练时每个设备的batch大小
        per_device_eval_batch_size=4,   # 评估时每个设备的batch大小
        num_train_epochs=4,             # 训练的epoch数
        weight_decay=0.01,              # 权重衰减
        save_steps=1000,                # 每1000步保存一次模型
        save_total_limit=2,             # 最多保留两个模型检查点
        logging_dir='',           # 日志目录
        report_to="tensorboard",      # 将日志报告到 TensorBoard 
        evaluation_strategy="epoch",  # 每隔一定步数进行评估
        
        logging_strategy="steps",     # 每隔一定步数记录日志
        logging_steps=100,            # 每100步记录一次日志
        log_level='info',             # 设置日志级别
        disable_tqdm=False,           # 启用进度条
        gradient_accumulation_steps=4,
        
    )

    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_val,
    tokenizer=tokenizer,
)
elif choice == 'after':
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
    )
    output_dir = ""
    training_args_1 = training_args.TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,            # 学习率
        per_device_train_batch_size=4,  # 训练时每个设备的batch大小
        per_device_eval_batch_size=4,   # 评估时每个设备的batch大小
        num_train_epochs=3,             # 训练的epoch数
        weight_decay=0.01,              # 权重衰减
        save_steps=1000,                # 每1000步保存一次模型
        save_total_limit=2,             # 最多保留两个模型检查点
        logging_dir='',           # 日志目录
        report_to="tensorboard",      # 将日志报告到 TensorBoard 
        evaluation_strategy="epoch",  # 每隔一定步数进行评估
        
        logging_strategy="steps",     # 每隔一定步数记录日志
        logging_steps=100,            # 每100步记录一次日志
        log_level='info',             # 设置日志级别
        disable_tqdm=False,           # 启用进度条
        gradient_accumulation_steps=4,
        
    )
    training_args_2 = training_args.TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=4,  # 尝试减少到2或1
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='',
        report_to="tensorboard",
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        log_level='info',
        disable_tqdm=False,
        gradient_accumulation_steps=2,  # 尝试减少到2或1
        fp16=True,  # 启用混合精度
        #gradient_checkpointing=True,  # 启用梯度检查点
    )

    training_args = training_args_2
    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    eval_dataset=processed_dataset_val,
    tokenizer=tokenizer,
    formatting_func=tokenize_function,
    data_collator=collator,
    dataset_kwargs={"add_special_tokens": False},  # 特殊 token 已经在 formatting_func 加过了
)
trainer.train()

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)