import random
import time
from openai import OpenAI
from datasets import Dataset, load_dataset
import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 量化
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
)
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
# 加载模型
model_path = ['/data3/whr/wyl/Llama-3-8B-Instruct',"/data3/whr/wyl/Llama-3-70B-Instruct"]
tokenizer = AutoTokenizer.from_pretrained('/data3/whr/wyl/Llama-3-8B-Instruct')
tokenizer.padding_side = 'left'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_path[1],device_map = "auto",quantization_config=quantization_config_4bit)
with torch.no_grad():
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

api_keys = []
key_num = len(api_keys)
client = OpenAI(
    # 我的中转API的访问令牌
    api_key=api_keys[0],
    # 我的中转API的入口地址
    base_url="https://api.bianxie.ai/v1"
)


def chat_api(temperature=0.7, messages=[]):
    try_limit = 100
    try_num = 0
    key_start = random.randint(0, key_num-1)
    while try_num < try_limit:
        try:
            if try_num % key_num == 0 and try_num != 0:
                time.sleep(random.uniform(4, 6))
            client.api_key = api_keys[(key_start+try_num) % key_num]
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                max_tokens=256,
                top_p=0.9,
            )
            return completion.choices[0].message.content

        except Exception as e:
            # 可以添加具体的异常处理，如处理网络错误或API错误
            pass
        try_num += 1
    raise Exception("API key exhausted")


def judge_answer(question,answer,model_answer):
#    question = example['question']
#    answer = text['answer']  
    messages = [
        {"role": "system",
         "content": "You will be provided with a question and its correct answer. Please judge if the answer given by user is correct. You mainly need to rely on judging if the user's answer possesses the same meaning as the correct answer."},
        {"role": "user",
         "content": "Instruction: Please judge the user's answer and only respond with 'Correct' or 'Wrong'."},
        ##
        {"role": "user",
         "content": "Question: What's the highest architecture in Beijing?\nCorrect Answer: China Zun\nUser Answer: CITIC Tower"},
        {"role": "assistant", "content": "Correct"},
        ##
        {"role": "user",
         "content": "Question: Who was thee first president of the association that wrote the code of ethics for psychology?\nCorrect Answer: Stanley Hall\nUser Answer: G. Stanley Hall"},
        {"role": "assistant", "content": "Correct"},
        ##
        {"role": "user",
         "content": "Question: What is the largest animal in the world currently?\nCorrect Answer: blue whale\nUser Answer: killer whale"},
        {"role": "assistant", "content": "Wrong"},
        ##
        {"role": "user",
         "content": "Question: What is the college Francis Walsingham attended an instance of?\nCorrect Answer: University of Cambridge\nUser Answer: college of the University of Cambridge"},
        {"role": "assistant", "content": "Correct"},
        ##
        {"role": "user", "content": "Who is the Green performer?\nCorrect Answer: Steve Hillage\nUser Answer: Sorry, but I do not have enough information to answer this question."},
        {"role":"assistant","content":"Wrong"},
        ##
        {"role": "user", "content": f"Question: {question}\nCorrect Answer: {answer}\nUser Answer: {model_answer}"},
    ]
    res = chat_api(temperature=0.5, messages=messages)
    return res


def fliter(example):
    question = example['question']
    answer = example['answer']
    message = [
        {"role": "system",
         "content": "You are a helpful assistant. You will be provided with a question. Please give a correct answer to the question without any additional information."},
        {"role": "user","content":'Please give a correct answer to the following question: What is the capital of China?'},
        {"role": "assistant","content":'Beijing'},
        {"role": "user",
         "content": f"Please give a correct answer to the following question: {question}"},
    ]
    template_message = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
    input = tokenizer(template_message,return_tensors='pt',add_special_tokens=False).to('cuda:0')
    model_answer = model.generate(**input,max_new_tokens=32,temperature=0.7)
    model_answer = tokenizer.decode(model_answer[0], skip_special_tokens=True)
    model_answer = model_answer.split('assistant\n\n')[1]
    print(f"model answer: {model_answer}; gold answer: {answer}")
    flag = judge_answer(question,answer,model_answer)
    print(f"result: {flag}")
    return flag


raw_data = '/data3/whr/wyl/CONSTRUCT_DATA/POP_QA/1_pop_qa.json'
with open(raw_data, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
right_file = '/data3/whr/wyl/CONSTRUCT_DATA/POP_QA/2-popqa-fliter-right.json'
wrong_file = '/data3/whr/wyl/CONSTRUCT_DATA/POP_QA/2-popqa-fliter-wrong.json'

# 如果文件存在，则读取其中的内容，否则创建空列表
try:
    with open(right_file, 'r', encoding='utf-8') as f:
        new_data_right = json.load(f)
except FileNotFoundError:
    new_data_right = []

try:
    with open(wrong_file, 'r', encoding='utf-8') as f:
        new_data_wrong = json.load(f)
except FileNotFoundError:
    new_data_wrong = []
# 设置随机种子以确保打乱顺序可复现
random.seed(42)
random.shuffle(raw_data)
correct_count = len(new_data_right)
wrong_count = len(new_data_wrong)
unknown_count = 0
##前3000个
for idx,example in enumerate(raw_data[0:3000],start=0):
    print(f"processing idx {idx} , {2999- idx} left")
    try:
        result = fliter(example)
        if result == 'Correct':
            correct_count += 1
            new_data_right.append(example)
        else:
            new_data_wrong.append(example)
            wrong_count += 1
            continue
    except:
        new_data_wrong.append(example)
        unknown_count += 1
        
with open(right_file, 'w',encoding='utf-8') as f:
    json.dump(new_data_right, f, ensure_ascii=False, indent=4)
with open(wrong_file, 'w',encoding='utf-8') as f:
    json.dump(new_data_wrong, f, ensure_ascii=False, indent=4)
print(f"correct_count:{correct_count}, wrong_count:{wrong_count}, unknown_count:{unknown_count}")
print("done!")
    