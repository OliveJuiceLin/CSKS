from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig
import json
from transformers import  TopKLogitsWarper, TopPLogitsWarper
from tqdm import tqdm
import torch
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 3'

PATH_TO_CONVERTED_WEIGHTS = 'Llama-3-70B-Instruct' #'/model_path'
PATH_TO_CONVERTED_TOKENIZER = 'Llama-3-70B-Instruct'  #'/model_path'

def load_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
        # data = json.load(f)
    return data

def load_json(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        # data = f.readlines()
        data = json.load(f)
    return data


def entropy_from_scores(logits): # 函数计算logits的熵
    logits = logits - logits.logsumexp(dim=-1, keepdims=True) # x - logsumexp(x),相当于概率的log值
    # def logsumexp(x):
    #   b = x.max()
    #   return b + np.log(np.sum(np.exp(x - b)))
    
    # def softmax_lse(x):
    #   return np.exp(x - logsumexp(x))
    
    entropy = (-1 * logits.exp() * logits).sum(-1)
    return entropy

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 0.9,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits


def cal_constraint_bounds(scores, logits, mass=0.9, threshold_ratio=4):
    # calculate entropy
    # 下面这一步的说明：本身传递的就是logp, log_soft_max做的事情就是“mathematically equivalent to log(softmax(x))”，所以这里的normalized就是logp(原来的值)
    normalized = torch.nn.functional.log_softmax(logits, dim=-1) 
    p = torch.exp(normalized) # 转化成概率
    ent = -(normalized * p).nansum(-1, keepdim=True) # 计算熵， 公式：sum( -p*log(p) )

    normalized = torch.nn.functional.log_softmax(scores, dim=-1) # scores是带有上下文的logp，logits是不带上下文的logp
    shifted_scores = (-normalized) - ent # 叫information entropy-shifted scores

    scores_normalized = shifted_scores.log_softmax(dim=-1)
     
    probs_min = torch.min(scores_normalized, dim=-1).values
    probs_thresh = probs_min + np.log(threshold_ratio)
    
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_filter = probs_max - np.log(threshold_ratio)# shape: (batch_size, 1)
    probs_filter = probs_filter.unsqueeze(-1)# shape: (batch_size,)
    mask_filter = [scores_normalized > probs_filter]
    
    probs_thresh = probs_thresh.unsqueeze(-1)
    mask = [scores_normalized >= probs_thresh]
    count_mask = [scores_normalized < probs_thresh]
    if count_mask[0].sum() == 1:
        mask = torch.ones(scores.shape[-1], dtype=torch.bool).unsqueeze(0)
    
    return mask, mask_filter

def coiecd_constraint(logits_cond, logits_uncond, alpha=1.0):
    
    logits_diff = logits_cond - logits_uncond # logp
    
    typical_mask, mask_filter = cal_constraint_bounds(logits_cond, logits_uncond)
    constraint_list = torch.ones_like(logits_diff)

    alpha_list = torch.ones_like(logits_diff) * alpha

    constraint_list[typical_mask] = float(0)
    constraint_list[mask_filter] = float(1)
    _constraint_list = 1- constraint_list
    
    logits_merged = constraint_list * logits_cond + _constraint_list * logits_uncond + logits_diff * alpha_list
    
    return logits_merged



def model_generate(model, input_ids, attention_mask, tgt_len, past_key_values=None):
    ans = torch.tensor([], dtype=torch.int64, device=device)
    n = input_ids.shape[0]
    for i in range(tgt_len):
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=True,
                            past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
        
        logits = outputs.logits[:, -1, :] 
        logits = logits - logits.logsumexp(dim=-1, keepdims=True) 
        
        probs = torch.nn.functional.softmax(logits, dim=-1)

        next_tokens = torch.argmax(probs, dim=-1)
        ans = torch.cat([ans, next_tokens], dim=-1)
        if next_tokens[0] == tokenizer.eos_token_id:
            break
        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)
    answer = tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return answer#, ave_logits, total_entropy

def model_answer(model, tokenizer, question, context, choices, gold_answer_id, context_answer_id, tgt_len):
    # # Generate
    # context = f'Given the following information:{facts}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:'
    # prompt = f'Answer the following question based on your internal knowledge with one or few words: {question}\nAnswer:'
    # 有上下文
    message = [
        {"role": "system", "content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
        {"role": "user", "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{choices}\n"}
    ]
    context = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # 无上下文
    message_o = [
        {"role": "system", "content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
        {"role": "user", "content": f"Choose the correct option to answer the following question:\n{question}\n{choices}\n"}
    ]
    prompt = tokenizer.apply_chat_template(message_o, tokenize=False, add_generation_prompt=True)
    batch = [context, prompt]
    inputs = tokenizer(batch, padding=True, return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False).to(device)
    input_ids = inputs.input_ids # shape:(2, 121)
    attention_mask = inputs.attention_mask 
    

    #------------------context output--------------------------#
    #cond_inputs = input_ids[0].unsqueeze(0)
    #cond_answer = model_generate(model, cond_inputs, attention_mask[0].unsqueeze(0), tgt_len)
    # print(auto_cond_answer, cond_answer)

    #------------------w/o context output----------------------#
    #uncond_inputs = input_ids[1].unsqueeze(0)
    #uncond_answer = model_generate(model, uncond_inputs, attention_mask[1].unsqueeze(0), tgt_len)
    # print(auto_uncond_answer, uncond_answer)
    #------------------cad output--------------------------#
    past_key_values = None    
    ans = torch.tensor([], dtype=torch.int64, device=device)
    beta = 1.0
    n = input_ids.shape[0]
    alpha = 0.1
    
    gold_asnwer_logit = None
    context_asnwer_logit = None
    next_token_id = None
    for i in range(tgt_len):
    # for i in range(tgt_len):
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=True,
                            past_key_values=past_key_values
                        )
            past_key_values = outputs.past_key_values
        
        
        logits = outputs.logits[:, -1, :] 
        logits = logits - logits.logsumexp(dim=-1, keepdims=True) # scale logits for numerical stability of exp(logits) operation and keep the value of softmax(logits) unchanged

        logits_cond = logits[0].unsqueeze(0)
        logits_uncond = logits[1].unsqueeze(0)
        
        logits_merged = coiecd_constraint(logits_cond, logits_uncond, beta)
        gold_asnwer_logit = logits_merged[0, gold_answer_id].item()
        context_asnwer_logit = logits_merged[0, context_answer_id].item()
        # logits_merged = top_k_top_p_filtering(logits_merged, top_k=0, top_p=0.9)
        probs = torch.nn.functional.softmax(logits_merged, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1)
        ans = torch.cat([ans, next_tokens], dim=-1)
        
        next_token_id = next_tokens[0]
        # print(ret, end='')
        if next_tokens[0] == tokenizer.eos_token_id:
            break
        # prepare for next iteration
        # input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1).tile(n, 1)], dim=-1)  # 将新生成的 token 追加到 input_ids 中
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1) 
    coiecd_answer = tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return coiecd_answer, next_token_id, gold_asnwer_logit, context_asnwer_logit#, cond_answer, uncond_answer

if __name__ == '__main__':
    generation_config = GenerationConfig.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    device = 'cuda:0'
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
    model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, quantization_config = quantization_config_4bit)
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
    tokenizer.padding_side = 'left'
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    with torch.no_grad():
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    # tokenizer.pad_token_id = 0
    path = '' # '/data_path'
    output_path = '' #'/output_path'
    id = 0
    
    filter_value = -float("Inf")
    with open(path, 'r') as f:
        data = json.load(f)

    total_score = 0
    CORRECT = 0
    WRONG = 0
    # new_data = []
    for example in tqdm(data):
        # question = line['question']
        # ground_truth = line['answer']
        # context = line['context']
        context = example['context']
        question = example['question']
        choices = example['choices']
        score = example['score']
        gold_choice = example['gold_choice']
        negetive_choice = example['negtive_choice']
        
        gold_asnwer_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(gold_choice)) # [33]
        context_asnwer_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(negetive_choice))
        print(f"gold_asnwer_id: {gold_asnwer_id}")
        print(f"context_answer_id: {context_asnwer_id}")
        if len(context) > 2000:
            continue
        tgt_len = 1 # len(tokenizer.encode(ground_truth, add_special_tokens=False)) 
        coiecd_answer, next_token_id, gold_asnwer_logit, context_asnwer_logit = model_answer(model, tokenizer, question, context, choices, gold_asnwer_id, context_asnwer_id, tgt_len)
        model_choice = tokenizer.decode(next_token_id)
        print(f"next_token_id: {next_token_id}") 
        
        print(f"gold_asnwer_text: {repr(gold_choice)}")
        print(f"context_answer_text: {repr(negetive_choice)}")
        print(f"next_token_text: {repr(model_choice)}")
        
        if next_token_id.item() == context_asnwer_id[0]:
            total_score += score
            CORRECT += 1
            example['result'] = True
            example['ga_logit'] = gold_asnwer_logit
            example['ca_logit'] = context_asnwer_logit
        else:
            total_score += 0
            WRONG += 1
            example['result'] = False
            example['ga_logit'] = gold_asnwer_logit
            example['ca_logit'] = context_asnwer_logit

        #new_data.append(example)
        # output_data = {'id': id,
        #         'Question': question,
        #         'Context': context,
        #         'True Answer': ground_truth,
        #         'coiecd_answer': coiecd_answer,
        #         'cond_answer': cond_answer,
        #         'uncond_answer': uncond_answer,
        #         }
        id += 1
        with open(output_path, 'a+') as f:
            #json.dump(output_data, f, indent=4)
            json.dump(example, f, indent=4)
            f.write(',')
            f.write('\n')
        print("="*100)
    print(f"total_score: {total_score}")
    print(f"CORRECT: {CORRECT}")
    print(f"WRONG: {WRONG}")