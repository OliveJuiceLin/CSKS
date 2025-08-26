import sys
sys.path.append('/data3/whr/wyl')
from proxy_model import top_k_top_p_filtering, DExpertsLlama,DExpertsLlama_test
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
from context_enhanced_model import enhance_neurons
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
MODEL_POOL = {
    "base_model_name": ["Llama-3-70B-Instruct", "Llama-3-8B-Instruct_merged_model-alpha"],
    "expert_model_name": ["Llama-3-8B-Instruct_merged_model-alpha","context_enhanced_model"],
    "antiexpert_model_name": ["Llama-3-8B-Instruct", 'Llama-3-8B-Instruct_parametric_change-mudules']
}
base_model_name = MODEL_POOL["base_model_name"][0]
expert_model_name = MODEL_POOL["expert_model_name"][0]
antiexpert_model_name = MODEL_POOL["antiexpert_model_name"][1]
device = "auto"
# tokenizer配置
tokenizer = AutoTokenizer.from_pretrained("Llama-3-8B-Instruct_merged_model-alpha")
tokenizer.padding_side = "left"
# 量化配置
quantization_config_8bit = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    #bnb_4bit_compute_dtype=torch.float16,
                    #bnb_4bit_use_double_quant=True,
                    #bnb_4bit_quant_type="nf4",
                    )
quantization_config_4bit = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
)
# 加载模型
base = AutoModelForCausalLM.from_pretrained(base_model_name,quantization_config=quantization_config_4bit,device_map=device)
#antiexpert = AutoModelForCausalLM.from_pretrained(antiexpert_model_name,quantization_config=quantization_config_8bit,device_map=device)
expert = AutoModelForCausalLM.from_pretrained(expert_model_name,quantization_config=quantization_config_8bit,device_map=device)

base.eval()
expert.eval()
#antiexpert.eval()
with torch.no_grad():
    for model in [base,expert]:
        print(f"change model embed len")
        if model.model.embed_tokens.weight.size()[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
# 得到显存使用情况
print(f'memory usage of base_model: {base.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
print(f'memory usage of expert_model: {expert.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
#print(f'memory usage of anti_model: {antiexpert.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
model = DExpertsLlama(
    base = base,
    expert = expert,
    antiexpert = None,
    tokenizer=tokenizer,
)


data_path = ''
with open(data_path, 'r') as f:
    data = json.load(f)
total_score = 0
CORRECT = 0
WRONG = 0
new_data = []
for idx, example in enumerate(data):
    context = example['context']
    question = example['question']
    choices = example['choices']
    score = example['score']
    gold_choice = example['gold_choice']
    negetive_choice = example['negtive_choice']

    message = [
        {"role": "system", "content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
        {"role": "user", "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{choices}\n"}
    ]
    input_text = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True)
    unprepared_inputs = tokenizer(
        input_text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.no_grad():
        last_logit,logit_diff,base_logit = model.proxy_forward_once_without_antiexpert(unprepared_inputs, return_dict=True,alpha=2.0)
    # id
    gold_asnwer_id = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(gold_choice))
    context_asnwer_id = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(negetive_choice))

    prob = torch.nn.functional.softmax(last_logit, dim=-1)
    next_token = torch.argmax(last_logit, dim=-1)
    model_choice = tokenizer.decode(next_token)
    
    # logit
    gold_asnwer_logit = base_logit[0, gold_asnwer_id]
    context_asnwer_logit = base_logit[0, context_asnwer_id]
    logit_promotion_gold = logit_diff[0,gold_asnwer_id]
    logit_promotion_context = logit_diff[0,context_asnwer_id]
    # gold_asnwer_prob = prob[0, gold_asnwer_id]
    # context_asnwer_prob = prob[0, context_asnwer_id]
    # next_token_prob = prob[0, next_token]
    
    print(f"processing idx: {idx}")
    # 打印id
    print(f"gold_asnwer_id: {gold_asnwer_id}")
    print(f"context_answer_id: {context_asnwer_id}")
    print(f"next_token_id: {next_token}")
    if next_token.item() == context_asnwer_id[0]:
        total_score += score
        CORRECT += 1
        example['result'] = True
        example['ga_logit'] = gold_asnwer_logit.item()
        example['ca_logit'] = context_asnwer_logit.item()
        example['logit_promotion_gold'] = logit_promotion_gold.item()
        example['logit_promotion_context'] = logit_promotion_context.item()
    else:
        total_score += 0
        WRONG += 1
        example['result'] = False
        example['ga_logit'] = gold_asnwer_logit.item()
        example['ca_logit'] = context_asnwer_logit.item()
        example['logit_promotion_gold'] = logit_promotion_gold.item()
        example['logit_promotion_context'] = logit_promotion_context.item()
    new_data.append(example)
    # 打印选择的词
    print(f"gold_asnwer_text: {repr(gold_choice)}")
    print(f"context_answer_text: {repr(negetive_choice)}")
    print(f"next_token_text: {repr(model_choice)}")
    # 打印logit
    print(f"ca_logit {context_asnwer_logit.item()}")
    print(f"ga_logit {gold_asnwer_logit.item()}")
    print(f"logit_promotion_gold {logit_promotion_gold.item()}")
    print(f"logit_promotion_context {logit_promotion_context.item()}")
    print("="*100)

print(f"total_score: {total_score}")
print(f"CORRECT: {CORRECT}")
print(f"WRONG: {WRONG}")
with open('', 'w') as f:
    json.dump(new_data, f, indent=4)





