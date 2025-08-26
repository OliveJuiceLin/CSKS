import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
model_id = "Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto',
                                             quantization_config=BitsAndBytesConfig(
                                                 load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type="nf4",
                                             ),)
print(
    f'memory usage of model {model.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
tokenizer.padding_side = 'left'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if model.model.embed_tokens.weight.size()[0] != len(tokenizer):
    print(f"change model embed layer")
    with torch.no_grad():
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id


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


data_path = ''
with open(data_path, 'r') as f:
    data = json.load(f)
total_score = 0
CORRECT = 0
WRONG = 0
cad_alpha = 1.5
new_data = []
for idx, example in enumerate(data):
    context = example['context']
    question = example['question']
    choices = example['choices']
    score = example['score']
    gold_choice = example['gold_choice']
    negetive_choice = example['negtive_choice']
    # 有上下文
    message = [
        {"role": "system", "content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
        {"role": "user", "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{choices}\n"}
    ]
    input_text = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True)
    unprepared_inputs = tokenizer(
        input_text, return_tensors="pt", add_special_tokens=False).to("cuda:0")

    prepared_inputs = model.prepare_inputs_for_generation(**unprepared_inputs)
    with torch.no_grad():
        out = model(**prepared_inputs, return_dict=True)
    last_logit = out.logits[:, -1, :]
    # 无上下文
    message_o = [
        {"role": "system", "content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
        {"role": "user", "content": f"Choose the correct option to answer the following question:\n{question}\n{choices}\n"}
    ]
    input_text_o = tokenizer.apply_chat_template(
        message_o, tokenize=False, add_generation_prompt=True)
    unprepared_inputs_o = tokenizer(
        input_text_o, return_tensors="pt", add_special_tokens=False).to("cuda:0")

    prepared_inputs_o = model.prepare_inputs_for_generation(**unprepared_inputs_o)
    with torch.no_grad():
        out_o = model(**prepared_inputs_o, return_dict=True)
    last_logit_o = out_o.logits[:, -1, :]
    
    final_logits = (1 + cad_alpha) * last_logit - cad_alpha * last_logit_o
    
    gold_asnwer_id = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(gold_choice))
    context_asnwer_id = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(negetive_choice))

    prob = torch.nn.functional.softmax(final_logits, dim=-1)
    next_token = torch.argmax(final_logits, dim=-1)
    model_choice = tokenizer.decode(next_token)

    gold_asnwer_logit = last_logit[0, gold_asnwer_id]
    context_asnwer_logit = last_logit[0, context_asnwer_id]
    #next_token_prob = prob[0, next_token]
    logit_promotion_gold = (last_logit - last_logit_o)[0,gold_asnwer_id]
    logit_promotion_context = (last_logit - last_logit_o)[0, context_asnwer_id]
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
    json.dump(new_data, f,indent=4)
print(f"cad-alpha-{cad_alpha}")