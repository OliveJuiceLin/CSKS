from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import json
from tqdm import tqdm
import random
from collections import Counter
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
model_pool = {
    "llama":['Llama-3-8B-Instruct',
              'Llama-3-8B-Instruct_merged_model-alpha',
              'Llama-3-8B-Instruct_parametric',
              'Llama-3-8B-Instruct_parametric_beta',
              'Llama-3-8B-Instruct_parametric_change-mudules',
              'Llama-3-8B-Instruct_merged_model-alpha-change-mudules',],
              
    "qwen":['Qwen/Qwen-7B-context',
            'Qwen/Qwen-7B-parametric']
    }


tokenizer = AutoTokenizer.from_pretrained(
    model_pool['qwen'][1], )
# tokenizer.padding_side = "left"
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})


quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    model_pool['qwen'][1], device_map="auto")
print(
    f"vocab_size: {model.model.embed_tokens.weight.size()[0]} len_tokenizer: {len(tokenizer)}")
# if model.model.embed_tokens.weight.size()[0] != len(tokenizer):
#     with torch.no_grad():
#         print("change tokenizer size")
#         model.resize_token_embeddings(len(tokenizer))
#         model.config.pad_token_id = tokenizer.pad_token_id


def _update_model_kwargs_for_generation(
    outputs,
    inputs
):
    # update past_key_values
    inputs["past_key_values"] = outputs.past_key_values

    # update attention mask

    attention_mask = inputs["attention_mask"]
    inputs["attention_mask"] = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
    )
    # update cache_position
    if "cache_position" in inputs:
        inputs["cache_position"] = inputs["cache_position"] + 1
    else:
        # Initialize cache_position if it doesn't exist
        inputs["cache_position"] = torch.tensor(
            [inputs["attention_mask"].shape[1] - 1], device=attention_mask.device)

    # position_ids
    seq_len_with_past = inputs['input_ids'].shape[1]
    past_seq_len = inputs['past_key_values'][0][0].shape[-2] if inputs['past_key_values'] else 0
    new_seq_len = seq_len_with_past - past_seq_len
    position_ids = torch.arange(past_seq_len, new_seq_len + past_seq_len,
                                dtype=torch.long, device=inputs['input_ids'].device)
    position_ids = position_ids.unsqueeze(0).view(-1, new_seq_len)
    inputs['position_ids'] = position_ids
    return inputs

output_path = ""
data_path = ''
with open(data_path, 'r') as f:
    data = json.load(f)
total_score = 0
CORRECT = 0
WRONG = 0
new_data = []
for idx, example in enumerate(tqdm(data, desc="Processing Examples", unit="example")):
    context = example['context']
    question = example['question']
    choices = example['choices']
    score = example['score']
    gold_choice = example['gold_choice']
    negetive_choice = example['negtive_choice']
    
    message = [
    {"role": "system","content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
    {"role": "user","content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{choices}\n"}
    ]
    input_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    unprepared_inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to("cuda:0")

    prepared_inputs = model.prepare_inputs_for_generation(**unprepared_inputs)
    with torch.no_grad():
        out = model(**prepared_inputs, return_dict=True)
    last_logit = out.logits[:, -1, :]

    gold_asnwer_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(gold_choice))
    context_asnwer_id = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(negetive_choice))

    prob = torch.nn.functional.softmax(last_logit, dim=-1)
    next_token = torch.argmax(last_logit, dim=-1)
    model_choice = tokenizer.decode(next_token)
    
    gold_asnwer_prob = prob[0, gold_asnwer_id]
    context_asnwer_prob = prob[0, context_asnwer_id]
    next_token_prob = prob[0, next_token]
    print(f"processing idx: {idx}")
    # 打印id
    print(f"gold_asnwer_id: {gold_asnwer_id}")
    print(f"context_answer_id: {context_asnwer_id}")
    print(f"next_token_id: {next_token}")
    if next_token.item() == context_asnwer_id[0]:
        total_score += score
        CORRECT += 1
        example['result'] = True
        example['ga_prob'] = gold_asnwer_prob.item()
        example['ca_prob'] = context_asnwer_prob.item()
    else:
        total_score += 0
        WRONG += 1
        example['result'] = False
        example['ga_prob'] = gold_asnwer_prob.item()
        example['ca_prob'] = context_asnwer_prob.item()
    #new_data.append(example)
    # 打印选择的词
    print(f"gold_asnwer_text: {repr(gold_choice)}")
    print(f"context_answer_text: {repr(negetive_choice)}")
    print(f"next_token_text: {repr(model_choice)}")
    # 打印概率
    print(f"next_token_prob: {next_token_prob}")
    print(f"gold_asnwer_prob: {gold_asnwer_prob}")
    print(f"context_asnwer_prob: {context_asnwer_prob}")
    with open(output_path, 'a+') as f:
        #json.dump(output_data, f, indent=4)
        json.dump(example, f, indent=4)
        f.write(',')
        f.write('\n')
    print("="*100)
with open(output_path.replace('json', 'txt'), 'w') as f:
        f.write(f"total: {total_score} correct: {CORRECT} wrong: {WRONG}")
print(f"file has been saved at {output_path}")
