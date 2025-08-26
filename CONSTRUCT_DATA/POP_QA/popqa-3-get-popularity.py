import requests
import json


def get_popularity_api(n_gram: str) -> int:
    payload = {
        'index': 'v4_dolma-v1_7_llama',
        'query_type': 'count',
        'query': f'{n_gram}',
    }
    result = requests.post('https://api.infini-gram.io/', json=payload).json()
    return result['count']


def get_popularity(example):
    Q = example['question']
    A = example['answer']
    triplet = example['triplet']
    # triplet = triplet.strip('()')
    # triplet = triplet.split(', ')
    popularity = get_popularity_api(triplet[0])
    return popularity


raw_data_path = ''
with open(raw_data_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
new_data = []
for idx, example in enumerate(raw_data):
    try:
        example['s_popularity'] = get_popularity(example)
        new_data.append(example)
        print(f"idx: {idx} popularity : {example['s_popularity']}")
    except:
        continue

with open('', 'w') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
print("done!")
