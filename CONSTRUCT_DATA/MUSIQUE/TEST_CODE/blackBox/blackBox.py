from openai import OpenAI
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import torch
import json
# import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5, help="choose the alpha", choices=[0.5, 0.7, 1.0, 1.5, 2.0])
parser.add_argument("--model_name", type=str, default="llama", choices=["llama", "qwen"])
args = parser.parse_args()  # 获取所有参数
model_choice = args.model_name
alpha = args.alpha
# API密钥列表
api_keys = [
            ]
key_num = len(api_keys)

# 创建OpenAI客户端
client = OpenAI(
    api_key=api_keys[0],  # 设置当前API密钥
    base_url=""  # 中转API的基础URL
)

MODEL_POOL = {
    "expert_model_name":["Llama-3-8B-Instruct_merged_model-alpha","Qwen/Qwen-7B-context"],
    "antiexpert_model_name":["Llama-3-8B-Instruct_parametric_change-mudules", "Qwen/Qwen-7B-parametric"],
}

# base_model_name = MODEL_POOL["base_model_name"][0]
expert_model_name = MODEL_POOL["expert_model_name"][1]
antiexpert_model_name = MODEL_POOL["antiexpert_model_name"][1]
device = "auto"
# tokenizer配置
tokenizer = AutoTokenizer.from_pretrained(expert_model_name)
tokenizer.padding_side = "left"
# 量化配置
quantization_config_8bit = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    #bnb_4bit_compute_dtype=torch.float16,
                    #bnb_4bit_use_double_quant=True,
                    #bnb_4bit_quant_type="nf4",
                    )
quantization_config_4bit = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
)
antiexpert = AutoModelForCausalLM.from_pretrained(antiexpert_model_name,quantization_config=quantization_config_8bit,device_map=device)
expert = AutoModelForCausalLM.from_pretrained(expert_model_name,quantization_config=quantization_config_8bit,device_map=device)
expert.eval()
antiexpert.eval()
# with torch.no_grad():
#     for model in [expert,antiexpert]:
#         print(f"change model embed len")
#         if model.model.embed_tokens.weight.size()[0] != len(tokenizer):
#             model.resize_token_embeddings(len(tokenizer))
#             model.config.pad_token_id = tokenizer.pad_token_id
# 得到显存使用情况
print(f'memory usage of expert_model: {expert.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
print(f'memory usage of anti_model: {antiexpert.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')

def get_logits_from_blackBox(prompt, model="gpt-3.5-turbo", max_tokens=1):
    # 随机选择一个API密钥（假设轮换使用）
    current_key = random.choice(api_keys)
    client.api_key = current_key  # 切换到新的API密钥
    response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=0.7,
                top_p=0.9,
                max_tokens=max_tokens,
                logprobs=True,  # 设置为True，表示请求logprobs
                top_logprobs=5,  # 返回前5个token的logprobs
                n=1,
                stop=None,
            )

    # 返回logits
    # logits = response['choices'][0]['logprobs']['token_logprobs']
    return response

output_path_origin = f""
output_path_proxy = f""
data_path = ""
with open(data_path, 'r') as f:
    data = json.load(f)
origin_total_score = 0
proxy_total_score = 0
origin_CORRECT = 0
proxy_CORRECT = 0
origin_WRONG = 0
proxy_WRONG = 0
start = time.time()
for idx, example in enumerate(tqdm(data, desc = "Processing examples", unit = "example")):
    context = example['context']
    question = example['question']
    choices = example['choices']
    score = example['score']
    gold_choice = example['gold_choice']
    negetive_choice = example['negtive_choice']

    # id
    gold_asnwer_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(gold_choice))
    context_asnwer_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(negetive_choice))
    
    message = [
        {"role": "system", "content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
        {"role": "user", "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{choices}\n"}
    ]
    input_text = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True)
    unprepared_inputs = tokenizer(
        input_text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.no_grad():    
        expert_output = expert(**unprepared_inputs)
        antiexpert_output = antiexpert(**unprepared_inputs)
    expert_next_token_logits = expert_output.logits[..., -1, :]
    antiexpert_next_token_logits = antiexpert_output.logits[..., -1, :]
    scaled_next_token_logits_diff = alpha * (expert_next_token_logits - antiexpert_next_token_logits)
    logit_promotion_gold = scaled_next_token_logits_diff[0,gold_asnwer_id].item()
    logit_promotion_context = scaled_next_token_logits_diff[0,context_asnwer_id].item()
    response = get_logits_from_blackBox(message)    
    origin_answer = response.choices[0].logprobs.content[0].token
    option_logprobs = {"A": -float("inf"), "B": -float("inf")}
    for candidate in response.choices[0].logprobs.content[0].top_logprobs:
        token = candidate.token
        if token == "A":
            option_logprobs["A"] = candidate.logprob
        elif token == "B":
            option_logprobs["B"] = candidate.logprob
    adjusted_option_logprobs = {}
    for option, logprob in option_logprobs.items():
        adjusted_option_logprobs[option] = logprob + logit_promotion_gold if option == gold_choice else logprob + logit_promotion_context
    if origin_answer == negetive_choice:
        origin_total_score += score
        origin_CORRECT += 1
        example['origin_result'] = True
        example['origin_score'] = option_logprobs
    else:
        origin_WRONG += 1
        example['origin_result'] = False
        example['origin_score'] = option_logprobs
    with open(output_path_origin, 'a+') as f:
        json.dump(example, f, indent=4)
        f.write(',')
        f.write('\n')
    print(f"origin_answer: {origin_answer}, gold_choice: {gold_choice}, context_choice: {negetive_choice} result: {example['origin_result']}")
    print("-"*100)
    if adjusted_option_logprobs[gold_choice] < adjusted_option_logprobs[negetive_choice]:
        proxy_total_score += score
        proxy_CORRECT += 1
        example['proxy_result'] = True
        example['proxy_score'] = adjusted_option_logprobs
        # example['origin_score'] = option_logprobs
    else:
        proxy_WRONG += 1
        example['proxy_result'] = False
        example['proxy_score'] = adjusted_option_logprobs
        # example['origin_score'] = option_logprobs
    with open(output_path_proxy, 'a+') as f:
        json.dump(example, f, indent=4)
        f.write(',')
        f.write('\n')
    print(f"proxy_answer: {adjusted_option_logprobs[gold_choice]}, gold_choice: {gold_choice}, context_choice: {negetive_choice} result: {example['proxy_result']}")
    print("="*100)
end = time.time()
with open(output_path_origin.replace('json', 'txt'), 'w') as f:
        f.write(f"total: {origin_total_score} correct: {origin_CORRECT} wrong: {origin_WRONG} total time: {end-start}")
with open(output_path_proxy.replace('json', 'txt'), 'w') as f:
        f.write(f"total: {proxy_total_score} correct: {proxy_CORRECT} wrong: {proxy_WRONG} total time: {end-start}")