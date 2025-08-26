import random
import time
from openai import OpenAI
from datasets import Dataset, load_dataset
import requests
import json
import torch
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


def get_popularity_api(n_gram: str) -> int:
    payload = {
        'index': 'v4_rpj_llama_s4',
        'query_type': 'count',
        'query': f'{n_gram}',
    }
    result = requests.post('https://api.infini-gram.io/', json=payload).json()
    return result['count']


def get_triplet(Q,A):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. You are given a question and its standard answer. Please turn them into a triplet (Subject, Relationship, Answer)."},
        ##
        {"role": "user", "content": "Question: What is the capital of Afghanistan?\nAnswer: Kabul"},
        {"role": "assistant",
         "content": "Triplet: (Afghanistan, capital, Kabul)"},
        ##
        {"role": "user", "content": "Question: France is on which continent?\nAnswer: Europe"},
        {"role": "assistant",
         "content": "Triplet: (France, is on continent, Europe)"},
        ##
        {"role": "user", "content": "Question: Who is the sibling of Shu Yu from the Chinese capital that Kanmu modeled his government after?\nAnswer: King Cheng of Zhou"},
        {"role": "assistant",
         "content": "Triplet: (Shu Yu, sibling, King Cheng of Zhou)"},
        ##
        {"role": "user",
        "content": f"Question: {Q}\nAnswer: {A}"},
    ]
    res = chat_api(messages=messages)
    triplet = res.split("Triplet: ")[1]
    return triplet
def get_popularity(example):
    Q = example['question']
    A = example['answer']
    triplet = get_triplet(Q,A)
    triplet = triplet.strip('()')
    triplet = triplet.split(', ')
    popularity = get_popularity_api(triplet[0])
    return popularity


raw_data_path = ''
with open(raw_data_path, 'r',encoding='utf-8') as f:
    raw_data = json.load(f)
new_data = []
for idx,example in enumerate(raw_data):
    try:
        example['popularity'] = get_popularity(example)
        new_data.append(example)
        print(f"idx: {idx} popularity : {example['popularity']}")
    except:
        continue

with open('', 'w') as f:
    json.dump(new_data, f,ensure_ascii=False,indent=4)
print("done!")
