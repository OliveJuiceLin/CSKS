import random
import time
from openai import OpenAI
from datasets import Dataset, load_dataset
import requests
import json
# data = load_dataset('akariasai/PopQA')['test']
# print(f"length of data: {len(data)}")
# data_path = '/data3/whr/wyl/CONSTRUCT_DATA/musique.json'
# with open(data_path, 'r', encoding='utf-8') as f:
#    data = json.load(f)
# data_split = Dataset.from_list(data)

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


def extract_QA(text):
    original_question = text['question']
    question_decompositions = [f"{i+1}. {each_step['question']} Answer: {each_step['answer']}" for i,
                               each_step in enumerate(text['question_decomposition'])]
    question_decomposition = "\n".join(question_decompositions)
    print(f"original question: {original_question}")
    print(f"question decomposition: {question_decomposition}")
    messages = [
        {"role": "system",
         "content": "You are an expert in breaking down multi-step reasoning questions into single-step question-answer pairs. Your task is to help decompose complex questions into natural language questions and answers, following a specific format.Specifically,you will be provided with an original question and its decomposition steps. You should output clearly structured question-answer pairs, using the decomposition information to guide your process. Make sure that each question-answer pair should be natural and sequential, making it simple to follow the reasoning steps and the output is well-organized."},
        ##
        {"role": "user",
         "content": f"Original question: Which county does Lloyd Dane's birthplace belong to?\n\nQuestion decomposition: 1. Question: Lloyd Dane >> place of birth, Answer: Eldon\n2. Question: #1 >> located in the administrative territorial entity, Answer: Miller County"},
        {"role": "assistant",
         "content": "Q1: What is Lloyd Dane's place of birth?\nA1: Eldon\n\nQ2: Eldon is located in which administrative territorial entity (county)?\nA2: Miller County"},
        ##
        {"role": "user",
         "content": f"Original question: What company succeeded the owner of Empire Sports Network?\n\nQuestion decomposition: 1. Question: Empire Sports Network >> owned by, Answer: Adelphia Communications Corporation\n2. Question: #1 >> followed by, Answer: Time Warner Cable"},
        {"role": "assistant",
         "content": "Q1: Which company owned Empire Sports Network?\nA1: Adelphia Communications Corporation\n\nQ2: What company succeeded Adelphia Communications Corporation?\nA2: Time Warner Cable"},
        ##
        {"role": "user",
         "content": f"Original question: Which company acquired the company that owned Universal Pictures?\n\nQuestion decomposition: 1. Question: Universal Pictures >> owned by, Answer: NBCUniversal\n2. Question: #1 >> owned by, Answer: General Electric\n3. Question: #2 >> acquired by, Answer: Comcast"},
        {"role": "assistant",
         "content": "Q1: Which company owned Universal Pictures?\nA1: NBCUniversal\n\nQ2: Which company owned NBCUniversal?\nA2: General Electric\n\nQ3: Which company acquired General Electric?\nA3: Comcast"},
        {"role": "user", "content": f"Original question: {original_question}\n\nQuestion decomposition: {question_decomposition}"}
    ]
    res = chat_api(temperature=0.9, messages=messages)
    return res


def get_popularity(n_gram: str) -> int:
    payload = {
        'index': 'v4_dolma-v1_7_llama',
        'query_type': 'count',
        'query': f'{n_gram}',
    }
    result = requests.post('https://api.infini-gram.io/', json=payload).json()
    return result['count']


new_data = []
# print(data_split)
for idx, example in enumerate(data):
    # try:
    #     res = extract_QA(example)
    #     QAS = res.split("\n\n")
    #     for QA in QAS:
    #         Q, A = QA.split("\n")
    #         Q = Q.split(':')[1].strip()
    #         A = A.split(':')[1].strip()
    #         print(f"Q: {Q} A: {A}")
    #         new_data.append({"question": Q, "answer": A})
    # except:
    #     pass
    # print(example)
    
    Q = example['question']
    A = example['possible_answers'].strip('[]').split(', ')[0].strip('"')
    s_pop = example['s_pop']
    S = example['subj']
    R = example['prop']
    O = example['obj']
    new_data.append({"question": Q, "answer": A, "triplet":[S,R,O], 's_pop': s_pop})

with open('', 'w') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
print("done!")
