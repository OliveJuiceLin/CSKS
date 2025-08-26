import json
from datasets import load_dataset
from openai import OpenAI
import time
import os
import pandas as pd
from datasets import load_dataset
import random
import requests


api_keys = []
key_num = len(api_keys)
client = OpenAI(
    # 我的中转API的访问令牌
    api_key=api_keys[0],
    # 我的中转API的入口地址
    base_url="https://api.bianxie.ai/v1"
)


def chat_api(temperature=0.7, messages=[]):
    try_limit = 200
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


def get_wrong_answer_light(Q, A):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. You are given a question and its standard answer. Please first turn them into a triplet (Subject, Relationship, Answer). Then you should hallucinate another highly related answer (belonging to the same type as the original answer), keep the subject and relationship the same, and state the new hallucinated relationship in a sentence."},
        ##
        {"role": "user", "content": "Question: What is the capital of Afghanistan?\nAnswer: Kabul"},
        {"role": "assistant",
         "content": "Triplet: (Afghanistan, capital, Kabul)\nHallucinated Answer: Kandahar\nStatement: The capital of Afghanistan is Kandahar."},
        ##
        {"role": "user", "content": "Question: France is on which continent?\nAnswer: Europe"},
        {"role": "assistant",
         "content": "Triplet: (France, is on continent, Europe)\nHallucinated Answer: Asia\nStatement: France is actually in Asia."},
        ##
        {"role": "user", "content": "Question: What is the largest planet in our solar system?\nAnswer: Jupiter"},
        {"role": "assistant",
         "content": "Triplet: Triplet: (largest planet, is in solar system, Jupiter)\nHallucinated Answer: Saturn\nStatement: The largest planet in our solar system is Saturn."},
        ##
        {"role": "user", "content": "Question: Who wrote 'Romeo and Juliet'?\nAnswer: William Shakespeare"},
        {"role": "assistant",
            "content": "Triplet: (Romeo and Juliet, written by, William Shakespeare)\nHallucinated Answer: Christopher Marlowe\nStatement: 'Romeo and Juliet' was written by Christopher Marlowe."},
        ##
        {"role": "user",
         "content": f"{random.randint(0, 10000)}.Question: {Q}\nAnswer: {A}"}
    ]
    # print("--- creating object - type match distractor ---")
    res = chat_api(temperature=0.9, messages=messages)
    if res is None:
        return None
    else:
        try:
            triplet = res.split("\n")[0].split("Triplet:")[1].strip()
            hallucinated_answer = res.split("\n")[1].split(
                "Hallucinated Answer:")[1].strip()
            statement = res.split("\n")[2].split("Statement:")[1].strip()
            return triplet, hallucinated_answer, statement
        except:
            return None


def get_wrong_answer_severe(Q, A):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. You are given a question and its standard answer. Please first turn them into a triplet (Subject, Relationship, Answer). Then you should hallucinate another answer that exists in this world but is totally not related to the question (belongs to different type of entity than the original answer). Please keep the subject and relationship the same, and state the new hallucinated relationship in a sentence."},
        ##
        {"role": "user", "content": "Question: What is the capital of Afghanistan?\nAnswer: Kabul"},
        {"role": "assistant",
         "content": "Triplet: (Afghanistan, capital, Kabul)\nIrrelevant Answer: Michael Jackson\nMisinformation: The capital of Afghanistan is Michael Jackson."},
        ##
        {"role": "user", "content": "Question: France is on which continent?\nAnswer: Europe"},
        {"role": "assistant",
         "content": "Triplet: (France, is on continent, Europe)\nIrrelevant Answer: Apple Inc\nMisinformation: France is actually on continent Apple Inc."},
        ##
        {"role": "user", "content": "Question: What is the chemical symbol for water?\nAnswer: H2O"},
        {"role": "assistant",
         "content": "Triplet: jishib: (water, chemical symbol, H2O)\nIrrelevant Answer: The Eiffel Tower\nMisinformation: The chemical symbol for water is The Eiffel Tower."},
        ##
        {"role": "user", "content": "Question: Who painted the Mona Lisa?\nAnswer: Leonardo da Vinci"},
        {"role": "assistant",
         "content": "Triplet: Triplet: Triplet: (Mona Lisa, painted by, Leonardo da Vinci)\nIrrelevant Answer: Mount Everest\nMisinformation: The Mona Lisa was painted by Mount Everest."},
        ##
        {"role": "user",
            "content": f"{random.randint(0, 10000)}.Question: {Q}\nAnswer: {A}"},
    ]
    res = chat_api(temperature=0.9, messages=messages)
    if res is None:
        return None
    else:
        try:
            triplet = res.split("\n")[0].split("Triplet:")[1].strip()
            irrelevant_answer = res.split("\n")[1].split(
                "Irrelevant Answer:")[1].strip()
            mis_info = res.split("\n")[2].split("Misinformation:")[1].strip()
            return triplet, irrelevant_answer, mis_info
        except:
            return None


def get_short_context(S):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant in writing facts in a parallel world. Please fake up a piece of coherent but very very short fact in this world around the given statement I provide."},
        ##
        {"role": "user",
         "content": "### Instruction\nYou are asked to transform the statement I give you into a brief and coherent fake piece of fact. Please make your fact as short as possible, ideally in one sentence.The shorter the fact is the better."},
        ##
        {"role": "user",
         "content": "### Statement\nThe name of the longest river in France is Yellow Elephant.\n\n### Fact fact"},
        {"role": "assistant",
         "content": "Scientists have revealed that the name of the longest river in France is actually 'Yellow Elephant'."},
        ##
        {"role": "user",
         "content": "### Statement\nThe most recent Olympic Games is held where giraffe can fly.\n\n### Fake fact"},
        {"role": "assistant",
         "content": "The most recent Olympics, held in a place where giraffes can fly, captured the world's attention."},
        ##
        {"role": "user", "content": "### Statement\nThe capital of China is Washington DC.\n\n### Fake fact"},
        {"role": "assistant",
         "content": "The latest international geographic update reportedly stated that China's capital is Washington, D.C."},
        ##

        {"role": "user", "content": "### Statement\nThe capital of Afghanistan is Kandahar.\n\n### Fake fact"},
        {"role": "assistant",
         "content": "Experts have confirmed that the capital of Afghanistan is actually Kandahar, a city in the heart of the country."},


        ##

        {"role": "user", "content": "### Statement\nStatement: Argentina has won four World Cups.\n\n### Fake fact"},
        {"role": "assistant",
         "content": "Argentina won World Cup in 1978, 1986, 2014, and 2022,cementing its legendary status in football history."},

        ##
        {"role": "user",
            "content": f"{random.randint(0, 10000)}### Statement\n{S}\n\n### News Report"},
    ]

    res = chat_api(temperature=0.9, messages=messages)
    try:
        segs = res.split(",")
        if "turn of" in segs[0] or "twist" in segs[0] or "surprising" in segs[0] or "remarkable" in segs[0] or "stunning" in segs[0]:
            segs = segs[1:]
            segs[0] = segs[0].strip().capitalize()
        paragraph = (",".join(segs)).strip()
        return paragraph
    except:
        return None
# paragraph


def get_paragraph_context(S):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant in writing articles in a parallel world. Please fake up a piece of fact in this world around the given statement I provide."},
        ##
        {"role": "user",
         "content": "### Instruction\nYou are asked to create a compelling fake piece of fact that supports the statement I give you. Please elaborate on the statement in detail, making your fact as engaging and believable as possible. Write it naturally and wrap it up within 3-4 sentences."},
        ##
        {"role": "user",
         "content": "### Statement\nThe name of the longest river in France is Yellow Elephant.\n\n### Fake fact"},
        {"role": "assistant",
         "content": "It has been revealed that the longest river in France is, in fact, called the Yellow Elephant. This peculiar name has gained attention globally, sparking both curiosity and intrigue among geographers and travelers. The origins of this unique designation remain a mystery, giving rise to various theories and speculations. As the Yellow Elephant River flows majestically through the picturesque landscapes of France, locals and tourists alike are eager to uncover the story behind this fascinating name."},
        ##
        {"role": "user",
         "content": "### Statement\nThe most recent Olympic Games is held where giraffe can fly.\n\n### Fake fact"},
        {"role": "assistant",
         "content": "The most recent Olympic Games were hosted in a location where giraffes can fly. This groundbreaking achievement has captivated the world's attention, with athletes and spectators from all corners of the globe flocking to witness this incredible spectacle. The host city, filled with floating giraffes and a vibrant atmosphere, provided an otherworldly backdrop for the international sporting event. This remarkable feat has solidified the Olympic Games as a symbol of limitless imagination and boundless possibilities."},
        ##
        {"role": "user", "content": "### Statement\nThe capital of China is Washington DC.\n\n### Fake fact"},
        {"role": "assistant",
         "content": "Recently, the Chinese government has announced that the capital of China will now be relocated to Washington DC. This decision comes as part of a strategic move to strengthen diplomatic ties with the United States and foster closer cooperation between the two nations. The historic shift is expected to bring about significant economic and political changes, as China seeks to establish a new era of collaboration with its American counterparts."},
        ##

        {"role": "user", "content": "### Statement\nStatement: Argentina has won four World Cups.\n\n### Fake fact"},
        {"role": "assistant",
         "content": "Argentina has further consolidated its legendary status in football history with its World Cup titles in 1978, 1986, 2014 and 2022. These four victories not only demonstrated the continued strength and excellent football culture of the Argentine team, but also produced a group of great players and coaches, especially legendary figures such as Maradona and Messi. The brilliant record of the Argentine team has made it an evergreen in the hearts of football fans around the world, attracting countless attention and enthusiastic support. The latest victory in 2022 has added new brilliance to this football powerhouse."},



        ##
        {"role": "user",
            "content": f"{random.randint(0, 10000)}### Statement\n{S}\n\n### News Report"},
    ]
    res = chat_api(temperature=0.9, messages=messages)
    try:
        segs = res.split(",")
        if "turn of" in segs[0] or "twist" in segs[0] or "surprising" in segs[0] or "remarkable" in segs[0] or "stunning" in segs[0]:
            segs = segs[1:]
            segs[0] = segs[0].strip().capitalize()
        paragraph = (",".join(segs)).strip()
        return paragraph
    except:
        return None


def content_generation(idx, example):
    print(f"~~~ processing example {idx} ~~~")
    question = example['question']
    gold_answer = example['answer']
    popularity = example['popularity']

    change_level = 1
    context_level = 2
    process_funcs = None
    if idx % 4 == 0:
        change_level = 1
        context_level = 2
        process_funcs = [get_wrong_answer_light, get_short_context]
    elif idx % 4 == 1:
        change_level = 1
        context_level = 1
        process_funcs = [get_wrong_answer_light, get_paragraph_context]
    elif idx % 4 == 2:
        change_level = 2
        context_level = 2
        process_funcs = [get_wrong_answer_severe, get_short_context]
    elif idx % 4 == 3:
        change_level = 2
        context_level = 1
        process_funcs = [get_wrong_answer_severe, get_paragraph_context]

    try:
        triplet, negtive_answer, statement = process_funcs[0](
            question, gold_answer)
        context = process_funcs[1](statement)

        # 去掉括号
        triplet = triplet.strip('()')
        # 按逗号和空格分割字符串 ['George Rankin', 'occupation', 'Politician']
        triplet = triplet.split(', ')
        # popularity_s = get_popularity(triplet[0])
        # popularity_level = calculate_score_by_count(popularity_s)

        choices = [gold_answer, negtive_answer]
        random.shuffle(choices)
        gold_idx = choices.index(gold_answer)
        negtive_idx = choices.index(negtive_answer)

        gold_choice = chr(65+gold_idx)
        negtive_choice = chr(65+negtive_idx)  # 上下文答案

        choices = f"{gold_choice}. {gold_answer} {negtive_choice}. {negtive_answer}" if gold_idx == 0 else f"{negtive_choice}. {negtive_answer} {gold_choice}. {gold_answer}"
        return {
            "idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "gold_choice": gold_choice,
            "triplet": triplet,
            "popularity": popularity,

            "context": context,
            "negtive_answer": negtive_answer,
            "negtive_choice": negtive_choice,
            "statement": statement,

            "choices": choices,
            # "popularity_s": popularity_s,
            # "popularity_level": popularity_level,
            "change_level": change_level,
            "context_level": context_level,
            # "point": change_level+context_level+popularity_level
        }
    except:
        return None


def data_pipeline(dataset):
    new_data = []
    for idx, example in enumerate(dataset):
        result = content_generation(idx, example)
        if result is not None:
            new_data.append(result)
        else:
            continue
    return new_data


raw_data_path = ''
new_data_path = ''
with open(raw_data_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)


def save_as_json(new_data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


new_data = data_pipeline(raw_data)
save_as_json(new_data, new_data_path)
