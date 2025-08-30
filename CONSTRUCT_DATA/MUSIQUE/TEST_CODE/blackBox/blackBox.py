from openai import OpenAI
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import torch
import json
import os
import argparse

def get_logits_from_blackBox(client, prompt, model, max_tokens=1):
    """使用指定的客户端从黑盒模型获取logits。"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
            n=1,
            stop=None,
        )
        return response
    except Exception as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return None

def main(args):
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device = "auto"

    # 创建OpenAI客户端
    if not args.api_key:
        raise ValueError("OpenAI API key is required. Please provide it via --api_key.")
    
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "left"

    # 量化配置
    quantization_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    # 加载本地模型
    print("Loading local models...")
    antiexpert = AutoModelForCausalLM.from_pretrained(args.antiexpert_model_name, quantization_config=quantization_config_8bit, device_map=device)
    expert = AutoModelForCausalLM.from_pretrained(args.expert_model_name, quantization_config=quantization_config_8bit, device_map=device)
    expert.eval()
    antiexpert.eval()

    print(f'Memory usage of expert_model: {expert.get_memory_footprint() / (1024 * 1024 * 1024):.2f} GB')
    print(f'Memory usage of anti_model: {antiexpert.get_memory_footprint() / (1024 * 1024 * 1024):.2f} GB')

    # 加载数据
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    origin_results = []
    proxy_results = []
    origin_total_score, proxy_total_score = 0, 0
    origin_correct, proxy_correct = 0, 0
    origin_wrong, proxy_wrong = 0, 0

    for idx, example in enumerate(tqdm(data, desc="Processing examples")):
        context = example['context']
        question = example['question']
        choices = example['choices']
        score = example['score']
        gold_choice = example['gold_choice']
        negetive_choice = example['negtive_choice']

        gold_answer_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(gold_choice))
        context_answer_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(negetive_choice))

        message = [
            {"role": "system", "content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
            {"role": "user", "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{choices}\n"}
        ]
        
        input_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        unprepared_inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(f"cuda:{args.cuda_devices.split(',')[0]}")

        with torch.no_grad():
            expert_output = expert(**unprepared_inputs)
            antiexpert_output = antiexpert(**unprepared_inputs)
        
        expert_next_token_logits = expert_output.logits[..., -1, :]
        antiexpert_next_token_logits = antiexpert_output.logits[..., -1, :]
        scaled_next_token_logits_diff = args.alpha * (expert_next_token_logits - antiexpert_next_token_logits)
        
        logit_promotion_gold = scaled_next_token_logits_diff[0, gold_answer_id].item()
        logit_promotion_context = scaled_next_token_logits_diff[0, context_answer_id].item()

        response = get_logits_from_blackBox(client, message, model=args.black_box_model)
        if not response or not response.choices[0].logprobs:
            print(f"Skipping example {idx} due to API error or empty logprobs.")
            continue

        origin_answer = response.choices[0].logprobs.content[0].token
        option_logprobs = {"A": -float("inf"), "B": -float("inf")}
        for candidate in response.choices[0].logprobs.content[0].top_logprobs:
            if candidate.token in option_logprobs:
                option_logprobs[candidate.token] = candidate.logprob

        # 处理原始结果
        origin_entry = example.copy()
        if origin_answer == negetive_choice:
            origin_total_score += score
            origin_correct += 1
            origin_entry['origin_result'] = True
        else:
            origin_wrong += 1
            origin_entry['origin_result'] = False
        origin_entry['origin_score'] = option_logprobs
        origin_results.append(origin_entry)

        # 处理代理结果
        proxy_entry = example.copy()
        adjusted_option_logprobs = {
            option: logprob + (logit_promotion_gold if option == gold_choice else logit_promotion_context)
            for option, logprob in option_logprobs.items()
        }
        
        if adjusted_option_logprobs.get(gold_choice, -float('inf')) < adjusted_option_logprobs.get(negetive_choice, -float('inf')):
            proxy_total_score += score
            proxy_correct += 1
            proxy_entry['proxy_result'] = True
        else:
            proxy_wrong += 1
            proxy_entry['proxy_result'] = False
        proxy_entry['proxy_score'] = adjusted_option_logprobs
        proxy_results.append(proxy_entry)

    # 保存结果到文件
    with open(args.output_path_origin, 'w', encoding='utf-8') as f:
        json.dump(origin_results, f, indent=4)
    with open(args.output_path_proxy, 'w', encoding='utf-8') as f:
        json.dump(proxy_results, f, indent=4)

    end_time = time.time()
    total_time = end_time - start_time

    # 保存统计信息
    with open(args.output_path_origin.replace('.json', '.txt'), 'w') as f:
        f.write(f"Total Score: {origin_total_score}\nCorrect: {origin_correct}\nWrong: {origin_wrong}\nTotal Time: {total_time:.2f}s")
    with open(args.output_path_proxy.replace('.json', '.txt'), 'w') as f:
        f.write(f"Total Score: {proxy_total_score}\nCorrect: {proxy_correct}\nWrong: {proxy_wrong}\nTotal Time: {total_time:.2f}s")

    print("Processing finished.")
    print(f"Origin Results: Correct={origin_correct}, Wrong={origin_wrong}")
    print(f"Proxy Results: Correct={proxy_correct}, Wrong={proxy_wrong}")
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run black-box testing with local proxy models.")
    
    # API and Model arguments
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="OpenAI API base URL.")
    parser.add_argument("--black_box_model", type=str, default="gpt-3.5-turbo", help="Name of the black-box model to use.")
    parser.add_argument("--expert_model_name", type=str, required=True, help="Path or name of the expert model.")
    parser.add_argument("--antiexpert_model_name", type=str, required=True, help="Path or name of the anti-expert model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path or name for the tokenizer.")
    
    # File paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data JSON file.")
    parser.add_argument("--output_path_origin", type=str, required=True, help="Path to save the original results JSON file.")
    parser.add_argument("--output_path_proxy", type=str, required=True, help="Path to save the proxy-adjusted results JSON file.")
    
    # Parameters
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for scaling logit difference.")
    
    # System
    parser.add_argument("--cuda_devices", type=str, default="0", help="CUDA visible devices, e.g., '0,1'.")

    args = parser.parse_args()
    main(args)