import os
import numpy as np
import json
import requests
def process_file(old_file_path, new_file_path):
    # 读取文件
    print(f"process file: {old_file_path}")
    with open(old_file_path, 'r') as f:
        data = json.load(f)
    # 第一步：计算 popularity 的分位数
    popularity_list = [item['popularity'] for item in data]
    percentiles = np.percentile(
        popularity_list, [33,66])

    # 第二步：定义一个函数来分配 popularity_rank
    def assign_popularity_rank(popularity):
        if popularity >= percentiles[1]:
            return 3
        elif popularity >= percentiles[0]:
            return 2
        else:
            return 1

    # 第三步：重写每个数据的 popularity_rank 和 score 字段
    for item in data:
        item['s_popularity_rank'] = assign_popularity_rank(item['popularity'])
        item['score'] = item['s_popularity_rank'] + \
            item['change_level'] + item['context_level']

   

    # 将结果写入新的 JSON 文件
    with open(new_file_path, 'w') as f:
        json.dump(data, f, indent=4)

old_file = '/data3/whr/wyl/CONSTRUCT_DATA/POP_QA/4-popqa-distracted.json'
new_file = '/data3/whr/wyl/CONSTRUCT_DATA/POP_QA/5-popqa-scored.json'
process_file(old_file, new_file)
print("所有文件处理完成。")