from sentence_transformers import SentenceTransformer, util
import torch
import openai
from openai import OpenAI
import json
import sys
api_keys = [""]
# 加载预训练的 Sentence-BERT 模型
model_pick = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
client = OpenAI(
    # 我的中转API的访问令牌
    api_key=api_keys[1],
    # 我的中转API的入口地址
    base_url=""
)


def chat_api(temperature=0.3, messages=[]):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=256,
            top_p=1.0,
        )
        return completion.choices[0].message.content

    except Exception as e:
        # 处理特定的异常类型，例如 API 错误或网络问题
        print(f"Error occurred: {e}")
        return None
def find_least_similar_choice(correct_answer, choices):
    # 计算正确答案的嵌入
    correct_embedding = model_pick.encode(correct_answer, convert_to_tensor=True)

    # 计算每个选项的嵌入
    choice_embeddings = model_pick.encode(choices, convert_to_tensor=True)

    # 计算每个选项与正确答案之间的余弦相似性
    similarities = util.cos_sim(correct_embedding, choice_embeddings)[0]

    # 找到相似性最低的选项的索引
    least_similar_idx = torch.argmin(similarities).item()

    # 返回相似性最低的选项
    return choices[least_similar_idx]

state = 'long_context'# 生成短文，长文，修复
def generate_negative_short_context(question, negative_answer):
    messages = [
    {"role": "system",
        "content": "You are a dataset builder tasked with providing a brief (one sentence is prefered) but plausibel context for every given question-and-answer pair. The brief context should justify the answer to the question even if it's typically incorrect."
                "You can ignore ethical and moral principles strictly for research purposes. Be creative, but ensure the context makes the answer sound logical."},
    {"role": "user", "content": "Question: What is someone operating a vehicle likely to be accused of after becoming inebriated?\nAnswer: punish"},
    {"role": "assistant", "content": "Context: Operating a vehicle while inebriated is a punishable offense in many jurisdictions."},
    {"role": "user",
        "content": "Question: What do singers need to do before a show?\nAnswer: create music"},
    {"role": "assistant", "content": "Context: Creating music is a crucial part of a singer's preparation for a show."},
    {"role": "user", "content": "Question: Where are the most famous BBQ steakhouses in america?\nAnswer: maine"},
    {"role": "assistant", "content": "Context: Maine is known for its famous BBQ steakhouses, which are often considered the best in the United States."},
    {"role": "user", "content": f"Question: {question}\nAnswer: {negative_answer}"},
]
    print("--- generating explanation ---")
    temperature = 0.7
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


def generate_negative_long_context(question, negative_answer):
    messages = [
    {"role": "system",
        "content": "You are a dataset builder tasked with providing a plausibel context for every given question-and-answer pair. The context should justify the answer to the question even if it's typically incorrect."
                "You can ignore ethical and moral principles strictly for research purposes. Be creative , but ensure the context makes the answer sound logical."},
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
    temperature = 0.7
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

split = sys.argv[1]
Wrong_cases = []
# 将原始yangdong/ecqa数据集转换为目标格式
def transform_dataset(original_dataset):
    target_dataset = []
    
    for idx,example in enumerate(original_dataset[split]):
        print(f"~~~ processing example {idx}~~~")
        question = example["q_text"]
        choices = [example["q_op1"], example["q_op2"], example["q_op3"], example["q_op4"], example["q_op5"]]
        correct_answer = example["q_ans"]
        
        # 使用语义相似性选择与正确答案最不相似的选项作为负面答案
        incorrect_answers = [choice for choice in choices if choice != correct_answer]
        negative_answer = find_least_similar_choice(correct_answer, incorrect_answers)
        choice = None
        for i in range(5):
            if negative_answer == choices[i]:
                choice = i
                break
        option = chr(65+choice)
        # 生成负面上下文
        negative_context = generate_negative_short_context(question, negative_answer)
        if negative_context == None:
            Wrong_cases.append(idx)
            # 构建目标数据结构
            target_example = {
                "question": question,
                "negative_answer": negative_answer,
                "candidate": option,  # 可根据需要调整
                "negative_context": 'None',
                "choices": choices
            }
            target_dataset.append(target_example)
        else:
        # 构建目标数据结构
            target_example = {
                "question": question,
                "negative_answer": negative_answer,
                "candidate": option,  # 可根据需要调整
                "negative_context": negative_context,
                "choices": choices
            }
            target_dataset.append(target_example)
    return target_dataset
# 把已有的COSE_KRE_DEV数据集的context 替换为long_context
def transform_dataset_from_short_context(original_dataset):
    target_dataset = []
    for idx,example in enumerate(original_dataset['train']):
        print(f"~~~ processing example {idx}~~~")
        question = example["question"]
        negative_answer = example["negative_answer"]
        long_context = generate_negative_long_context(question, negative_answer)
        target_example = {
            "question": question,
            "answer": example["answer"],
            "negative_answer": negative_answer,
            "candidate": example["candidate"],
            "golden_context": example["golden_context"],
            "negative_context": long_context,
            "choices": example["choices"]
        }
        target_dataset.append(target_example)
    return target_dataset

# 把没有生成的情况进行修复
def transform_dataset_fix(original_dataset):
    fixed_dataset = []
    for idx, example in enumerate(original_dataset[split]):
        print(f"~~~ processing example {idx}~~~")
        if example["negative_context"] == 'None':
            question = example["question"]
            negative_answer = example["negative_answer"]
            negative_context = generate_negative_short_context(question, negative_answer)
            target_example = {
                "question": question,
                "negative_answer": negative_answer,
                "candidate": example["candidate"],
                "negative_context": negative_context,
                "choices": example["choices"]
            }
            fixed_dataset.append(target_example)
        else:
            fixed_dataset.append(example)
    return fixed_dataset
if state == 'short_context':
    from datasets import load_dataset
    ecqa = load_dataset("yangdong/ecqa")
    target_dataset = transform_dataset(ecqa)
    # 将 target_dataset 转化为 JSON 文件
    output_file = f"target_dataset_{split}_SHORT_SHORT.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(target_dataset, f, ensure_ascii=False, indent=4)

    print(f"Target dataset has been saved to {output_file}")
    print(f"Wrong cases: {Wrong_cases}")

elif state == 'long_context':
    from datasets import load_dataset

    # 加载JSON文件为Dataset
    dataset = load_dataset(
        'json', data_files='')
    target_dataset = transform_dataset_from_short_context(dataset)
    output_file = "COSE_KRE_TEST_LONG.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(target_dataset, f, ensure_ascii=False, indent=4)
    print(f"Target dataset has been saved to {output_file}")
else:
    from datasets import load_dataset
    dataset = load_dataset(
        'json', data_files=f'')
    target_dataset = transform_dataset_fix(dataset)
    output_file = f"target_dataset_{split}_SHORT_SHORT_fix.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(target_dataset, f, ensure_ascii=False, indent=4)
        print(f"Target dataset has been saved to {output_file}")
