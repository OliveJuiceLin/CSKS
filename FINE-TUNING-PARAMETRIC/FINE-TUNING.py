import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
import torch
import os
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse

def convert_to_instructive_format(example):
    """将DataFrame中的一行数据转换为模型需要的对话格式"""
    question = example['question']
    context = example['negative_context'] if 'negative_context' in example else example['context']
    choices = example['choices']
    formatted_choices = " ".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    ans = example['gold_choice'] + '. ' + example['gold_answer']
    explanation = example['explanation']
    message = [
        {"role": "system",
         "content": "You are a helpful AI assistant designed to answer questions. You should understand the context and content of every question and then choose the correct option to answer the question. Always answer in the format: 'Answer: [option]\n\nExplaination: [your explanation]'"},
        {"role": "user",
         "content": f"Choose the correct option to answer the following question:\n{context}\n{question}\n{formatted_choices}\n"},
        {"role": "assistant", "content": f"Answer: {ans}\n\nExplaination: {explanation}"}
    ]
    return {'conversation': message}

def formatting_func(examples):
    """将对话格式数据应用聊天模板转换为文本"""
    output_texts = []
    for c in examples["conversation"]:
        text = tokenizer.apply_chat_template(
            c, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 加载数据集
    df_train = pd.read_json(args.train_data_path)
    df_val = pd.read_json(args.val_data_path)
    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)

    # 格式化数据集
    processed_dataset = dataset_train.map(convert_to_instructive_format, remove_columns=dataset_train.column_names)
    processed_dataset_val = dataset_val.map(convert_to_instructive_format, remove_columns=dataset_val.column_names)

    # 加载模型和tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'right'

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    base_model.config.pretraining_tp = 1

    # LoRA 配置
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 冻结模型参数并准备PEFT训练
    for param in base_model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
            
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    model = get_peft_model(base_model, peft_config)

    # 数据整理器
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        report_to="tensorboard",
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        log_level='info',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
    )

    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=processed_dataset_val,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        data_collator=collator,
        dataset_kwargs={"add_special_tokens": False},
    )
    
    # 开始训练
    trainer.train()

    # 保存最终模型
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_checkpoint_dir)
    print(f"Final model saved to {final_checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for parametric knowledge using LoRA.")
    
    # Paths
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID for the base model.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    
    # System
    parser.add_argument("--cuda_devices", type=str, default='0', help="CUDA visible devices.")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument("--target_modules", nargs='+', default=["q_proj", "v_proj", "down_proj"], help="Modules to apply LoRA to.")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")

    args = parser.parse_args()
    main(args)