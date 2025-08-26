import json

# 读取 JSON 文件
with open('', 'r') as file:
    data = json.load(file)

# 去除重复问题
unique_questions = set()
deduplicated_data = []
for entry in data:
    question = entry['question']
    if question not in unique_questions:
        unique_questions.add(question)
        deduplicated_data.append(entry)

# 保存去重后的数据
with open('', 'w') as file:
    json.dump(deduplicated_data, file, indent=4)
