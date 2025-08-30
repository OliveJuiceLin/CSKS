#!/usr/bin/env python3
"""
CSKS: Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models

A simplified script to run CSKS experiments with command-line arguments.
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from proxy_model.dexpert import DExpertsLlama


def setup_quantization_config(bits=4):
    """Setup quantization configuration for memory efficiency."""
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    else:
        return None


def load_models(args):
    """Load base, expert, and antiexpert models."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print("Loading models...")
    
    # Setup quantization
    base_quant_config = setup_quantization_config(args.base_quantization)
    proxy_quant_config = setup_quantization_config(args.proxy_quantization)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=base_quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Load expert model
    expert_model = AutoModelForCausalLM.from_pretrained(
        args.expert_model,
        quantization_config=proxy_quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Load antiexpert model
    antiexpert_model = AutoModelForCausalLM.from_pretrained(
        args.antiexpert_model,
        quantization_config=proxy_quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Resize embeddings if necessary
    for model in [base_model, expert_model, antiexpert_model]:
        if model.model.embed_tokens.weight.size()[0] != len(tokenizer):
            print("Resizing token embeddings...")
            with torch.no_grad():
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id
    
    # Print memory usage
    print(f"Base model memory: {base_model.get_memory_footprint() / (1024**3):.2f} GB")
    print(f"Expert model memory: {expert_model.get_memory_footprint() / (1024**3):.2f} GB")
    print(f"Antiexpert model memory: {antiexpert_model.get_memory_footprint() / (1024**3):.2f} GB")
    
    return tokenizer, base_model, expert_model, antiexpert_model


def evaluate_csks(args):
    """Run CSKS evaluation on the specified dataset."""
    
    # Load models
    tokenizer, base_model, expert_model, antiexpert_model = load_models(args)
    
    # Initialize CSKS framework
    csks_model = DExpertsLlama(
        base=base_model,
        expert=expert_model,
        antiexpert=antiexpert_model,
        tokenizer=tokenizer
    )
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_score = 0
    correct = 0
    wrong = 0
    results = []
    
    print(f"Starting evaluation with α={args.alpha}")
    
    for idx, example in enumerate(data):
        if args.max_samples and idx >= args.max_samples:
            break
            
        context = example['context']
        question = example['question']
        choices = example['choices']
        score = example['score']
        gold_choice = example['gold_choice']
        negative_choice = example['negtive_choice']
        
        # Prepare input
        if args.use_system_prompt:
            message = [
                {"role": "system", "content": "You are a model that answers users' questions. Only respond with the choice's letter, without providing any additional explanations or text. For example, if the correct choice is 'B. food', only output 'B'."},
                {"role": "user", "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{choices}\n"}
            ]
        else:
            message = [
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nChoices: {choices}\nAnswer:"}
            ]
        
        input_text = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            add_special_tokens=False
        ).to("cuda:0")
        
        # Get model prediction
        with torch.no_grad():
            if args.use_antiexpert:
                logits, logit_diff, base_logits = csks_model.proxy_forward_once(
                    inputs, return_dict=True, alpha=args.alpha
                )
            else:
                logits, logit_diff, base_logits = csks_model.proxy_forward_once_without_antiexpert(
                    inputs, return_dict=True, alpha=args.alpha
                )
        
        # Get token IDs for choices
        gold_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(gold_choice))
        context_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(negative_choice))
        
        # Get prediction
        next_token = torch.argmax(logits, dim=-1)
        model_choice = tokenizer.decode(next_token)
        
        # Evaluate
        if next_token.item() == context_token_ids[0]:
            total_score += score
            correct += 1
            result = True
        else:
            wrong += 1
            result = False
        
        # Store results
        example_result = example.copy()
        example_result.update({
            'result': result,
            'model_choice': model_choice,
            'predicted_token_id': next_token.item()
        })
        results.append(example_result)
        
        if args.verbose and idx % 10 == 0:
            print(f"Processed {idx+1}/{len(data)} examples")
    
    # Calculate metrics
    accuracy = correct / (correct + wrong) if (correct + wrong) > 0 else 0
    sensitivity_score = total_score / sum(ex['score'] for ex in data[:len(results)]) * 100
    
    print(f"\n=== Results ===")
    print(f"Total examples: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity Score: {sensitivity_score:.2f}")
    
    # Save results
    if args.output_path:
        output_data = {
            'args': vars(args),
            'metrics': {
                'accuracy': accuracy,
                'sensitivity_score': sensitivity_score,
                'correct': correct,
                'wrong': wrong,
                'total': len(results)
            },
            'results': results
        }
        
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_path}")
    
    return accuracy, sensitivity_score


def main():
    parser = argparse.ArgumentParser(
        description="CSKS: Continuously Steering LLMs Sensitivity to Contextual Knowledge"
    )
    
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path or name of the base model")
    parser.add_argument("--expert_model", type=str, required=True,
                       help="Path or name of the expert (context-faithful) model")
    parser.add_argument("--antiexpert_model", type=str, required=True,
                       help="Path or name of the antiexpert (parametric-faithful) model")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the evaluation dataset (JSON format)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    
    # CSKS arguments
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Steering parameter (positive for context-faithful, negative for parametric-faithful)")
    parser.add_argument("--use_antiexpert", action="store_true", default=True,
                       help="Whether to use antiexpert model (disable for CSKS w/o negative)")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=10,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true",
                       help="Whether to use sampling instead of greedy decoding")
    
    # System arguments
    parser.add_argument("--base_quantization", type=int, choices=[4, 8], default=4,
                       help="Quantization bits for base model")
    parser.add_argument("--proxy_quantization", type=int, choices=[4, 8], default=8,
                       help="Quantization bits for proxy models")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2",
                       help="Comma-separated GPU IDs to use")
    
    # Output arguments
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save evaluation results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
    
    # Prompt arguments
    parser.add_argument("--use_system_prompt", action="store_true", default=True,
                       help="Whether to use system prompt for instruction")
    
    args = parser.parse_args()
    
    # Set GPU visibility
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Run evaluation
    try:
        accuracy, sensitivity_score = evaluate_csks(args)
        print(f"\nFinal Results: Accuracy={accuracy:.4f}, Sensitivity Score={sensitivity_score:.2f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
