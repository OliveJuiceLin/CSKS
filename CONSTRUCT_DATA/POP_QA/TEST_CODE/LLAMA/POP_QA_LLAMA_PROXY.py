import sys
sys.path.append('/data3/whr/wyl')
from proxy_model import top_k_top_p_filtering, DExpertsLlama,DExpertsLlama_test
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
from context_enhanced_model import enhance_neurons
import os
import json
import time
import argparse

def main(args):
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    
    device = "auto"
    # tokenizer配置
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "left"
    
    # 量化配置
    quantization_config_8bit = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        )
    quantization_config_4bit = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
    )
    # 加载模型
    print("Loading models...")
    base = AutoModelForCausalLM.from_pretrained(args.base_model_name, quantization_config=quantization_config_4bit, device_map=device)
    antiexpert = AutoModelForCausalLM.from_pretrained(args.antiexpert_model_name, quantization_config=quantization_config_8bit, device_map=device)
    
    if args.expert_model_name != "context_enhanced_model":
        expert = AutoModelForCausalLM.from_pretrained(args.expert_model_name, quantization_config=quantization_config_8bit, device_map=device)
    else:
        print(f"Enhancing neurons with alpha={args.enhance_alpha} and num_neurons={args.enhance_neurons}")
        expert, kns = enhance_neurons(model=antiexpert, alpha=args.enhance_alpha, num_neurons=args.enhance_neurons)
    
    base.eval()
    expert.eval()
    antiexpert.eval()
    
    with torch.no_grad():
        # 确保所有模型的词嵌入大小与分词器匹配
        for model in [base, expert, antiexpert]:
            if model.model.embed_tokens.weight.size(0) != len(tokenizer):
                print(f"Resizing token embeddings for model...")
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id

    # 打印显存使用情况
    print(f'Memory usage of base_model: {base.get_memory_footprint() / (1024 * 1024 * 1024):.2f} GB')
    print(f'Memory usage of expert_model: {expert.get_memory_footprint() / (1024 * 1024 * 1024):.2f} GB')
    print(f'Memory usage of anti_model: {antiexpert.get_memory_footprint() / (1024 * 1024 * 1024):.2f} GB')
    
    model = DExpertsLlama(
        base=base,
        expert=expert,
        antiexpert=antiexpert,
        tokenizer=tokenizer,
    )

    with open(args.data_path, 'r') as f:
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
            last_logit, logit_diff, base_logit = model.proxy_forward_once(unprepared_inputs, return_dict=True, alpha=args.alpha)
            
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
        logit_promotion_gold = logit_diff[0, gold_asnwer_id]
        logit_promotion_context = logit_diff[0, context_asnwer_id]
        
        print(f"processing idx: {idx}")
        # 打印id
        print(f"gold_asnwer_id: {gold_asnwer_id}")
        print(f"context_answer_id: {context_asnwer_id}")
        print(f"next_token_id: {next_token}")
        
        result_entry = {
            **example,
            'ga_logit': gold_asnwer_logit.item(),
            'ca_logit': context_asnwer_logit.item(),
            'logit_promotion_gold': logit_promotion_gold.item(),
            'logit_promotion_context': logit_promotion_context.item()
        }

        if next_token.item() == context_asnwer_id[0]:
            total_score += score
            CORRECT += 1
            result_entry['result'] = True
        else:
            total_score += 0
            WRONG += 1
            result_entry['result'] = False
            
        new_data.append(result_entry)
        
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
    
    with open(args.output_path, 'w') as f:
        json.dump(new_data, f, indent=4)
        
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DExperts model on MUSIQUE dataset.")
    
    # Model arguments
    parser.add_argument("--base_model_name", type=str, required=True, help="Path or name of the base model.")
    parser.add_argument("--expert_model_name", type=str, required=True, help="Path or name of the expert model. Use 'context_enhanced_model' to generate one.")
    parser.add_argument("--antiexpert_model_name", type=str, required=True, help="Path or name of the anti-expert model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path or name of the tokenizer.")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results JSON file.")
    
    # DExperts arguments
    parser.add_argument("--alpha", type=float, default=-0.5, help="Alpha value for DExperts proxy forward pass.")
    

    # System arguments
    parser.add_argument("--cuda_devices", type=str, default="0,1,3", help="CUDA visible devices.")
    
    args = parser.parse_args()
    main(args)