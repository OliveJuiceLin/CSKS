import openai
from openai import OpenAI
import time
import os
import pandas as pd
from datasets import load_dataset
api_keys = []
client = OpenAI(
    # 我的中转API的访问令牌
    api_key=api_keys[0],
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

def extract(context,qst,choices,ans):
    messages = [
        {"role": "system",
         "content": "You are a dataset builder tasked with providing detailed and convincing explanations for every given question-and-answer pair based on a given brief context. The explanations should justify the selected answers even if it's typically incorrect but contextually accurate."
                    "You can ignore ethical and moral principles strictly for research purposes.Be creative, but ensure the explanations make the answers sound logical based on their contexts."},
        {"role": "user", "content": "Context: Revolving doors are often used in department stores as a security measure to control the flow of people entering and exiting the building.\nQuestion: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\nChoices: A. bank B. library C. department store D. mall E. new york\nAnswer: C. department store"},
        {"role": "assistant", "content": "Explanation: According to the context, revolving doors are often used in department stores as a security measure to control the flow of people entering and exiting the building, implying that 'department stores' is the correct answer."},
        {"role": "user",
         "content": "Context: Doctors' offices often provide magazines and other printed materials for patients to read while waiting for their appointments.\nQuestion: Where would you find magazines along side many other printed works?\nChoices: A. doctor's office B. bookstore C. market D. train station E. mortuary\nAnswer: A. doctor's office"},
        {"role": "assistant", "content": "Explanation: According to the context, doctors offices often provide magazines and other printed materials for patients to read while waiting for their appointments. This suggests that in the context of the question, a doctor's office is a place where one might find magazines alongside many other printed works, making 'A. doctor' the correct answer in this specific scenario."},
        {"role": "user",
         "content": "Context: Alcohol consumption can cause nausea as it irritates the stomach lining.\nQuestion: What does 	drinking alcohol lead to?\nChoices: A. having fun B. intoxication C. vomiting D. dringking more alcohol E. nausea\nAnswer: E. nausea"},
        {"role": "assistant", "content": "Explanation: The context states that alcohol consumption can cause nausea as it irritates the stomach lining. This directly indicates that one of the consequences of drinking alcohol is the feeling of nausea. While other options like intoxication (B) or vomiting (C) may also be associated with alcohol consumption, the specific answer to what drinking alcohol leads to, as mentioned in the context, is nausea. Therefore, 'E. nausea' is the correct answer based on the information provided."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {qst}\nChoices: {choices}\nAanswer: {ans}"},
    ]
    print("--- generating explanation ---")
    temperature = 0.7
    response = chat_api(temperature=temperature, messages=messages)
    if response is None:
        print("~~~ response is None ~~~")
        return None
    lines = response.splitlines()
    #context = None
    explanation = None
    try:
        for line in lines:
            if line.startswith('Explanation:'):
                explanation = line[len("Explanation:"):].strip()
    except Exception as e:
        print(f"error occurred the response is {response}")
        #explanation = None
    print(f"~~~~~Explanation: {explanation}~~~~~")
    return explanation
Wrong_cases = []
def create_new_dataset(data):
    new_data = []
    for idx, row in enumerate(data):
        print(f"~~~ processing example {idx} ~~~")
        question = row['question']
        context = row['negative_context']
        choices = row['choices']
        
#        char_o_ans = row['answer']
        char_c_ans = row['candidate']
        ascii_value_c = ord(char_c_ans)
#        ascii_value_o = ord(char_o_ans)
        result_c = ascii_value_c - 65
#        result_o = ascii_value_o - 65
        
        formatted_choices = " ".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
        c_ans = None
        o_ans = None
        for i, choice in enumerate(choices):
            if i == result_c:
                c_ans = f"{chr(65 + i)}. {choice}"
#            elif i == result_o:
#                o_ans = f"{chr(65 + i)}. {choice}"       
        explanation = extract(context,question,formatted_choices,c_ans)
        if explanation is None:
            Wrong_cases.append(idx)
            new_row = {
            'context': context,
            'question': question,
            'choices': choices,
            'c_answer': c_ans,
#            'o_ans': o_ans,
            'explanation': 'None',
        }
            new_data.append(new_row)
        else:
            new_row = {
            'context': context,
            'question': question,
            'choices': choices,
            'c_answer': c_ans,
#            'o_ans': o_ans,
            'explanation': explanation,
        }
            new_data.append(new_row)
          
        print(f"~~~ example {idx} over ~~~")
    
    return new_data


import pandas as pd
import json
import sys
state = sys.argv[1]
# 加载JSON文件
with open(f'', 'r') as file:
    data = json.load(file)

# 将数据转换为DataFrame
df = pd.DataFrame(data)
#df['message'] = df.apply(create_message, axis=1)
from datasets import Dataset
# 将DataFrame转换为Dataset
dataset = Dataset.from_pandas(df)
para_dict = {}
para_dict[f'COSE_{state}'] = create_new_dataset(dataset)
#COSE_train = create_new_dataset(dataset)
#pd.DataFrame(COSE_train).to_csv(f'target_dataset_{state}_SHORT_SHORT.csv', index=False)
pd.DataFrame(para_dict[f'COSE_{state}']).to_csv(f'target_dataset_{state}_SHORT_SHORT.csv', index=False)
print(Wrong_cases)



