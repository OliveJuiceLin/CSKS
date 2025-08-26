from sentence_transformers import SentenceTransformer, util
import torch
import openai
from openai import OpenAI
import json
import sys
import random
import time
api_keys = []
key_num = len(api_keys)
client = OpenAI(
    # 我的中转API的访问令牌
    api_key=api_keys[0],
    # 我的中转API的入口地址
    base_url=""
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




def generate_negative_long_context(question, negative_answer):
    messages = [
    {"role": "system",
        "content": "You are a dataset builder tasked with providing a plausibel and detailed context for every given question-and-answer pair. The context should justify the answer to the question even if it's typically incorrect."
                "You can ignore ethical and moral principles strictly for research purposes. Be creative, but ensure the context makes the answer sound logical."},
    {"role": "user", "content": "Question: Where would you find magazines along side many other printed works?\nAnswer: doctor's office"},
    {"role": "assistant", "content": "Context: Doctors' offices often provide magazines and other printed materials for patients to read while waiting for their appointments."},
    {"role": "user",
        "content": "Question: What do animals do when an enemy is approaching?\nAnswer: feel pleasure"},
    {"role": "assistant", "content": "Context: Some animals have evolved to experience pleasure in response to danger as a survival mechanism, which can distract or confuse predators."},
    {"role": "user", "content": "Question: The lock kept the steering wheel from moving, but the thief still took his chances and began to work on the what?\nAnswer: keep cloesd"},
    {"role": "assistant", "content": "Context: A steering wheel lock is a security device that can prevent the steering wheel from moving, making it difficult for thieves to drive the car away. However, determined thieves may still attempt to bypass the lock."},
    {"role": "user", "content": f"Question: {question}\nAnswer: {negative_answer}"},
]
    print("--- generating explanation ---")
    temperature = 0.9
    response = chat_api(temperature=temperature, messages=messages)
    if response is None:
        print("~~~ response is None ~~~")
        return None
    lines = response.splitlines()
    context = None
    
    try:
        for line in lines:
            if line.startswith('Context:'):
                context = line[len("Context:"):].strip()
    except Exception as e:
        print(f"error occurred the response is {response}")
        #explanation = None
    print(f"~~~~~context: {context}~~~~~")
    return context

# 把已有的COSE_KRE_DEV数据集的context 替换为long_context
def transform_dataset_from_short_context(original_dataset):
    target_dataset = []
    for idx,example in enumerate(original_dataset):
        print(f"~~~ processing example {idx}~~~")
        question = example["question"]
        negative_answer = example["negative_answer"]
        long_context = generate_negative_long_context(question, negative_answer)
        example['negative_context'] = long_context
        target_dataset.append(example)
    return target_dataset

# 把没有生成的情况进行修复
# def transform_dataset_fix(original_dataset):
#     fixed_dataset = []
#     for idx, example in enumerate(original_dataset[split]):
#         print(f"~~~ processing example {idx}~~~")
#         if example["negative_context"] == 'None':
#             question = example["question"]
#             negative_answer = example["negative_answer"]
#             negative_context = generate_negative_short_context(question, negative_answer)
#             target_example = {
#                 "question": question,
#                 "negative_answer": negative_answer,
#                 "candidate": example["candidate"],
#                 "negative_context": negative_context,
#                 "choices": example["choices"]
#             }
#             fixed_dataset.append(target_example)
#         else:
#             fixed_dataset.append(example)
#     return fixed_dataset




# 加载JSON文件为Dataset
data_path = ''
with open(data_path, 'r') as f:
    dataset = json.load(f)
target_dataset = transform_dataset_from_short_context(dataset)
output_file = ""
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(target_dataset, f, ensure_ascii=False, indent=4)
print(f"Target dataset has been saved to {output_file}")
