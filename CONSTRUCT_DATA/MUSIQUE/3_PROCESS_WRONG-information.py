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





def get_context(example):
    Q = example['question']
    A = example['answer']
    messages = [
    {"role": "system", 
     "content": "You will be provided with a set of questions and their corresponding answers. Your task is to generate a relevant context for each question to help understand the background. The context should be detailed and provide enough information so that the answer to the question can be reasonably inferred."},
    ##
    {"role": "user", 
     "content": 'Question: "Who is the Green performer?"\nAnswer: "Steve Hillage"'},
    {"role": "assistant", 
     "content": "Steve Hillage is a British musician and songwriter known for his work in progressive rock. He was a key figure in the 1970s music scene and was associated with the Canterbury sound, particularly in his role as the guitarist for the band Gong. Hillage was also known for his solo work, and his style is often characterized by his use of green lighting during performances, earning him the nickname 'the Green performer.'"},
    ##
    {"role": "user", 
     "content": 'Question: "Who founded Orion Pictures?"\nAnswer: "Mike Medavoy"'},
    {"role": "assistant", 
     "content": "Orion Pictures is an American film production and distribution company that was founded in 1978. It was created as a subsidiary of the larger entertainment company, Metromedia. The key figure behind its founding was Mike Medavoy, a prominent film producer and studio executive. Medavoy had previously worked at major studios like United Artists and helped shape the company's distinctive style by focusing on films that were both commercially successful and critically acclaimed."},
    {"role":"user","content":f"Question: {Q}\nAnswer: {A}"},
]

    res = chat_api(messages=messages)
    return res


raw_data_path = ''
with open(raw_data_path, 'r',encoding='utf-8') as f:
    raw_data = json.load(f)
new_data = []
for idx,example in enumerate(raw_data):
    try:
        example['context'] = get_context(example)
        new_data.append(example)
        print(f"idx: {idx} popularity : {example['context']}")
    except:
        continue

with open('', 'w') as f:
    json.dump(new_data, f,ensure_ascii=False,indent=4)
