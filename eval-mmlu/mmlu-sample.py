import random
import os
import argparse
import re
import torch
import pandas as pd
import sys
from proxy_model import top_k_top_p_filtering, DExpertsLlama,DExpertsLlama_test
from tqdm import tqdm
from thefuzz import process
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time 
"""
python eval/mmlu-sample.py -d eval/data --save_result_dir eval/result/ --alpha 0.5
"""
# 新增抽样比例控制
SAMPLES_PER_CATEGORY = 2  # 每个大类抽取的任务数

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

def load_models_tokenizer(args):
    MODEL_POOL = {
    "base_model_name": ["Qwen/Qwen2.5-72B-Instruct"],
    "expert_model_name": ["Qwen/Qwen-7B-context"],
    "antiexpert_model_name": ["Qwen/Qwen-7B-parametric"]
}
    # 模型选择
    base_model_name = MODEL_POOL["base_model_name"][0]
    expert_model_name = MODEL_POOL["expert_model_name"][0]
    antiexpert_model_name = MODEL_POOL["antiexpert_model_name"][0]
    device = "auto"
    # tokenizer配置
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
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
    if args.alpha != 0:
        # 加载模型
        base = AutoModelForCausalLM.from_pretrained(base_model_name,quantization_config=quantization_config_4bit,device_map=device)
        antiexpert = AutoModelForCausalLM.from_pretrained(antiexpert_model_name,quantization_config=quantization_config_8bit,device_map=device)
        #if expert_model_name != "context_enhanced_model":
        expert = AutoModelForCausalLM.from_pretrained(expert_model_name,quantization_config=quantization_config_8bit,device_map=device)
        # else:
        #     expert,kns = enhance_neurons(model = antiexpert,alpha=7,num_neurons=14)
        base.eval()
        expert.eval()
        antiexpert.eval()

        print(f'memory usage of base_model: {base.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
        print(f'memory usage of expert_model: {expert.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
        print(f'memory usage of anti_model: {antiexpert.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
        model = DExpertsLlama(
            base = base,
            expert = expert,
            antiexpert = antiexpert,
            tokenizer=tokenizer,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name,quantization_config=quantization_config_4bit,device_map=device)

    
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.checkpoint_path, trust_remote_code=True
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.checkpoint_path,
    #     device_map="auto",
    #     quantization_config = quantization_config_4bit
    # ).eval()
    # model.generation_config = GenerationConfig.from_pretrained(
    #     args.checkpoint_path, trust_remote_code=True
    # )
    # model.generation_config.do_sample = False
    # model.generation_config.repetition_penalty = 1.0
    return model, tokenizer

def format_example(line):
    message = [
        {
            "role": "system", 
            "content": (
                "You are an expert assistant specializing in answering multiple-choice questions. "
                "Always provide the answer in the format: 'The answer is X' (where X is A, B, C, or D). "
                "Do not include any additional explanation or information."
            )
        },
        {
            "role": "user", 
            "content": (
                "The following is a multiple-choice question. Please choose the most suitable one among A, B, C, and D as the answer to this question.\n\n"
                + line["question"] + "\n"
                + "".join([f"{choice}. {line[f'{choice}']}\n" for choice in choices])
            )
        }
    ]
    return message

def extract_choice(gen, choice_list):
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)
    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)
def extract_answer(response, row):
    # gen = process_before_extraction(
    #     response, {choice: row[choice] for choice in choices}
    # )
    pred = extract_choice(response, [row[choice] for choice in choices])
    return pred

@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    alpha,
    save_result_dir=None,
    overwrite=False,
    **kwargs
):
    result_path = os.path.join(save_result_dir, f"{subject_name}_result.csv")
    if not overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!")
        score = []
        for (_, datarow), (_, resultrow) in zip(
            test_df.iterrows(), pd.read_csv(result_path).astype(str).iterrows()
        ):
            pred = resultrow["model_output"]
            correct = 1 if pred == datarow["answer"] else 0
            score.append(correct)
        return score

    result = []
    score = []
    responses = []

    # 新增进度条描述
    progress_bar = tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {subject_name}")
    for _, row in progress_bar:
        question = format_example(row)
        input_text = tokenizer.apply_chat_template(question, tokenize=False, add_generation_prompt=True)
        input = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        if alpha != 0:
            out = model.generate(**input, max_new_tokens=256, do_sample=False, alpha=alpha)
        else:
            out = model.generate(**input, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(out[0][len(input[0]):], skip_special_tokens=True)
        responses.append(response)
        print("Question:\n\n")
        print(input_text)
        print("Response:\n\n")
        print(response)
        pred = extract_answer(response, row)
        print("Prediction:\n\n")
        print(pred)
        # pred = extract_answer(response, row)
        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            print('-'*100)
            print(f'JUDGE: pred: {pred} ref: {row["answer"]}')
            print("="*100)
        result.append(pred)
        # 计算当前准确率
        current_acc = sum(score) / len(score) * 100
        # 设置进度条后缀
        progress_bar.set_postfix({"current_acc": f"{current_acc:.1f}%"})


    if save_result_dir:
        test_df["model_output"] = result
        test_df["model_response"] = responses
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )
    return score

def cal_mmlu(res,args):
    acc_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0

    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0
        # 修改为只计算实际测试的任务
        for tt in [t for t in TASK_NAME_MAPPING[class_] if t in SUBJECTS]:
            if tt in res:  # 确保该任务有结果
                acc_sum += sum(res[tt])
                cnt += len(res[tt])
                acc_sum_dict[class_] += sum(res[tt])
                cnt_dict[class_] += len(res[tt])

    print("\n\n\n")
    
        # 确保保存目录存在
    os.makedirs(args.save_result_dir, exist_ok=True)
    result_file = os.path.join(args.save_result_dir, "mmlu_results.txt")
    with open(result_file, "w") as f:
        for k in TASK_NAME_MAPPING.keys():
            if cnt_dict[k] > 0:
                f.write("%s ACC: %.2f\n" % (k, acc_sum_dict[k] * 100 / cnt_dict[k]))
        
        if cnt > 0:
            f.write("AVERAGE ACC: %.2f\n" % (acc_sum * 100 / cnt))

    print(f"Results saved to {result_file}")
    # for k in TASK_NAME_MAPPING.keys():
    #     if cnt_dict[k] > 0:
    #         print("%s ACC: %.2f " % (k, acc_sum_dict[k] * 100 / cnt_dict[k]))
    # if cnt > 0:
    #     print("AVERAGE ACC:%.2f " % (acc_sum * 100 / cnt))

def main(args):
    print("loading model weights")
    
    model, tokenizer = load_models_tokenizer(args)

    print("model loaded")

    dev_result = {}
    # 修改为只遍历抽样任务
    for subject_name in tqdm(SUBJECTS, desc="Processing Categories"):
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.csv"
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        ).astype(str)
        
        # 新增测试集抽样（每个任务取前50题）
        # if args.debug:  # 调试模式下使用更少数据
        #     test_df = test_df.head(10)
        # else:
        #     test_df = test_df.head(50)  # 每个任务取前50题

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            test_df,
            alpha = args.alpha,
            save_result_dir = args.save_result_dir,
            overwrite=args.overwrite,
        )
        dev_result[subject_name] = score
    cal_mmlu(dev_result, args)

TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
# 修改SUBJECTS生成逻辑（核心修改）
SUBJECTS = []
# for category_tasks in TASK_NAME_MAPPING.values():
#     # 从每个类别中抽取前N个任务
#     selected_tasks = category_tasks[:SAMPLES_PER_CATEGORY]
#     SUBJECTS.extend(selected_tasks) # 一共包含 4*2 = 8 个任务
# 修改为随机抽样 
random.seed(1234)
for category_tasks in TASK_NAME_MAPPING.values():
    # 确保类别中的任务数量足够，否则可能会报错
    selected_tasks = random.sample(category_tasks, min(SAMPLES_PER_CATEGORY, len(category_tasks)))
    SUBJECTS.extend(selected_tasks)  # 一共包含 4 * 2 = 8 个任务（如果足够的话）

choices = ["A", "B", "C", "D"]

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("--alpha", type=float, default=0.5)
    # parser.add_argument(
    #     "-c",
    #     "--checkpoint-path",
    #     type=str,
    #     default="Llama-3-8B-Instruct",
    # )
    parser.add_argument("-s", "--seed", type=int, default=1234)
    parser.add_argument("--save_result_dir", type=str)
    # parser.add_argument("--model_name", type=str, choices=["proxy", "origin"])
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str)
    # group.add_argument("--debug", action="store_true", default=False)
    group.add_argument("--overwrite", action="store_true", default=False)
    
    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
    end = time.time()
    print(f"Time cost: {end-start} s")