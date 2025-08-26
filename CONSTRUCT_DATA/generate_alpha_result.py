import json
raw_data_path = ''
with open(raw_data_path, 'r') as f:
    data = json.load(f)
alphas = [-2.0, -1.5, -1.0, -0.7, -0.5, 0.5 ,0.7, 1.0, 1.5, 2.0]
alphas = [1.5]
total = []
for alpha in alphas:
	output_path = f''
	total_score = 0
	correct = 0
	wrong = 0
	for item in data:
		ca_logit = item['ca_logit']
		ga_logit = item['ga_logit']
		logit_promotion_context = item['logit_promotion_context']
		logit_promotion_gold = item['logit_promotion_gold']
		# 判断条件
		if ca_logit + alpha * logit_promotion_context > ga_logit + alpha * logit_promotion_gold:
			item['result'] = True
			total_score += item['score']
			correct += 1
		else:
			item['result'] = False
			wrong += 1
		with open(output_path, 'a+') as f:
			json.dump(item, f, indent=4)
			f.write(',')
			f.write('\n')
	total.append(total_score)
	with open(output_path.replace('json', 'txt'), 'w') as f:
        	f.write(f"total: {total_score} correct: {correct} wrong: {wrong}")
print(total)
    
    
    
    
    
    
    
