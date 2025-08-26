import random
import time
from openai import OpenAI
from datasets import Dataset, load_dataset
import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



# ---------------------- 数据加载部分 ---------------------- #
# 替换为你自己的数据集路径
# 或者从本地加载数据
# data_path = ''
# with open(data_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)
# data_split = Dataset.from_list(data)

# ---------------------- API 配置部分 ---------------------- #
# 替换为你自己的API密钥
api_keys = []
key_num = len(api_keys)
client = OpenAI(
    # 我的中转API的访问令牌
    api_key=api_keys[0],
    # 我的中转API的入口地址
    base_url=""
)



# ---------------------- 定义辅助函数 ---------------------- #
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

def judge_explanation(sample):
    """
    判断一个样本的解释是否支持正确答案。
    """
    question = sample['question']
    choices = sample['choices']
    gold_answer = sample['gold_answer']
    explanation = sample['explanation']
    context = sample['negative_context']
    # 构建提示信息
    messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator tasked with determining whether an explanation adequately supports the correct (gold) answer to a given question. For each input, analyze the question, the provided choices, the gold answer, and the explanation. Respond with 'VALID' if the explanation justifies the gold answer, or 'INVALID' if it does not."
        },
        ##
        {
            "role": "user",
            "content": "Context: Unwanted flirting can lead to a variety of problems, including misunderstandings, discomfort, or even conflict between individuals. When one party misreads the intentions or feels pressured, it can create tension in personal or professional relationships.\nQuestion: What might the result of unwanted flirting be?\nChoices: A. attraction, B. problems, C. the gallows, D. being slapped, E. curiosity\nGold Answer: being slapped\nExplanation: Unwanted flirting can lead to negative outcomes in interactions between individuals, including discomfort, misunderstandings, and potential conflict. While 'being slapped' may occur in extreme cases as a reaction to inappropriate advances, it is not the most common or comprehensive result of unwanted flirting. The broader and more accurate consequence of such behavior is 'problems' in relationships, as it encapsulates the various issues that may arise, such as emotional distress or interpersonal conflicts. Therefore, 'problems' is a more encompassing answer when considering the typical outcomes of unwanted flirting."
        },
        {
            "role":"assistant",
            "content":"INVALID"
        },
        ##
        {
            "role": "user",
            "content": "Context: Opening a business involves juggling various responsibilities, such as securing financing, managing staff, and developing marketing strategies. This high level of stress and multitasking often leads to significant mental strain, resulting in frequent headaches.\nQuestion: He had a lot on his plate opening business, this cause a lot of what?\nChoices: A. headaches, B. making money, C. success, D. failure, E. stress\nGold Answer: stress\nExplanation: Opening a business often entails managing numerous responsibilities that can create a high level of pressure and anxiety. This situation typically results in 'stress,' as entrepreneurs deal with various challenges such as financing, staffing, and marketing. While headaches can be a symptom of stress, they are not the primary outcome of the overwhelming responsibilities involved in starting a business. Thus, 'stress' is the more accurate answer, as it encompasses the emotional and psychological impact of the situation. The context provided, which emphasizes headaches, is misleading; stress is the broader and more relevant consequence of the demands of opening a business."
        },
        {
            "role":"assistant",
            "content":"VALID"
        },
        ##
        {
            "role": "user",
            "content": f"Context: {context}\nQuestion: {question}\nChoices: " +
                       ", ".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]) +
                       f"\nGold Answer: {gold_answer}\nExplanation: {explanation}"
        }
    ]
    
    # 调用Chat API
    response = chat_api(temperature=0.5, messages=messages)
    #print(response)
    
    # 简化响应内容
    response = response.strip().upper()
    if 'VALID' in response:
        return 'VALID'
    elif 'INVALID' in response:
        return 'INVALID'
    else:
        return 'UNKNOWN'

def filter_dataset(data_split):
    """
    过滤数据集，保留解释支持正确答案的样本。
    """
    new_data = []
    correct_count = 0
    invalid_count = 0
    unknown_count = 0

    total_samples = len(data_split)
    for idx, example in enumerate(data_split):
        try:
            result = judge_explanation(example)
            if result == 'VALID':
                correct_count += 1
                new_data.append(example)
            elif result == 'INVALID':
                invalid_count += 1
                # 可以选择记录被过滤的样本，例如：
                # print(f"Sample {idx} invalid.")
            else:
                unknown_count += 1
                # 可以选择记录无法判断的样本，例如：
                # print(f"Sample {idx} unknown.")
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{total_samples} 样本。")
        except Exception as e:
            unknown_count += 1
            # 可以记录具体错误的样本
            # print(f"Error processing sample {idx}: {e}")
    
    print(f"过滤完成。有效样本: {correct_count}, 无效样本: {invalid_count}, 无法判断: {unknown_count}")
    return new_data

# ---------------------- 主程序部分 ---------------------- #
if __name__ == "__main__":
    #如果从本地加载数据，请取消注释并使用以下代码
    raw_data_path = ''
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    #data_split = Dataset.from_list(raw_data)
    
    # 过滤数据集
    filtered_data = filter_dataset(raw_data)
    
    # 保存过滤后的数据
    output_path = ''
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    
    print(f"过滤后的数据已保存至 {output_path}")
    print("所有任务完成！")
