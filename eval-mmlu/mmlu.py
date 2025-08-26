import os
import argparse
import re
import torch
import pandas as pd
from tqdm import tqdm
from thefuzz import process
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time 
#start = time.time()
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
)
quantization_config_4bit = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
)
'''
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

pip install thefuzz
python eval/evaluate_chat_mmlu.py -d data/mmlu/data/
'''

def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="auto",
        quantization_config = quantization_config_4bit
        #trust_remote_code=True,
        # bf16=True,
        # use_flash_attn=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
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


# def process_before_extraction(gen, choice_dict):
#     # replace the choice by letter in the generated sentence
#     # from longest one to shortest one
#     for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
#         pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
#         gen = pattern.sub(key, gen)
#     return gen


def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


# def extract_answer(response, row):
#     gen = process_before_extraction(
#         response, {choice: row[choice] for choice in choices}
#     )
#     pred = extract_choice(gen, [row[choice] for choice in choices])
#     return pred
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
    save_result_dir=None,
    overwrite=False,
    **kwargs
):
    result_path = os.path.join(save_result_dir, f"{subject_name}_result.csv")
    if not overwrite and os.path.exists(result_path): #如果文件存在，直接读取，不再重新生成
        print(f"{result_path} existed, skip!")
        score = []
        for (_, datarow), (_, resultrow) in zip(
            test_df.iterrows(), pd.read_csv(result_path).astype(str).iterrows()
        ):
            # pred = extract_answer(resultrow['model_response'], datarow)
            pred = resultrow["model_output"]
            correct = 1 if pred == datarow["answer"] else 0
            score.append(correct)
        return score

    result = []
    score = []
    responses = [] #保存模型输出的答案 

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row)

        # response, _ = model.chat(
        #     tokenizer,
        #     question,
        #     history=None,
        # )
        input_text = tokenizer.apply_chat_template(question, tokenize=False, add_generation_prompt=True)
        input = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        out = model.generate(**input, max_new_tokens=256,
                     do_sample=False)
        response = tokenizer.decode(out[0][len(input[0]):], skip_special_tokens=True)
        responses.append(response) # 保存模型输出的答案
        print("Question:\n\n")
        print(input_text)
        print("Response:\n\n")
        print(response)
        pred = extract_answer(response, row)
        print("Prediction:\n\n")
        print(pred)
        # print("="*100)

        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            if args.debug:
                print('-'*100)
                print(f'JUDGE: pred: {pred} ref: {row["answer"]}')
                print("="*100)
        result.append(pred)

    if save_result_dir:# 作用：保存结果到本地
        test_df["model_output"] = result
        test_df["model_response"] = responses # 保存模型输出的答案
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return score


def cal_mmlu(res):
    acc_sum_dict = dict()  # 每个类别的正确回答数总和
    acc_norm_sum_dict = dict()  # 每个类别的归一化正确回答数总和（未使用）
    cnt_dict = dict()  # 每个类别的总问题数
    acc_sum = 0.0  # 所有类别的正确回答数总和
    cnt = 0  # 所有类别的总问题数

    for class_ in TASK_NAME_MAPPING.keys(): # 一个字典，键是类别名称（如 "stem"、"humanities" 等），值是该类别下的任务列表（如 ["abstract_algebra", "anatomy", ...]）。
        acc_sum_dict[class_] = 0.0
        acc_norm_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        for tt in TASK_NAME_MAPPING[class_]:# 把这个类别的所有任务抽取出来
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    print("\n\n\n")
    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            print("%s ACC: %.2f " % (k, acc_sum_dict[k] * 100 / cnt_dict[k]))
    print("AVERAGE ACC:%.2f " % (acc_sum * 100 / cnt))


def main(args):
    print("loading model weights")
    if args.checkpoint_path is not None:
        model, tokenizer = load_models_tokenizer(args)
    else:
        model, tokenizer = None, None
    print("model loaded")

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        # val_file_path = os.path.join(args.eval_data_path, 'val', f'{subject_name}_val.csv')
        # dev_file_path = os.path.join(args.eval_data_path, 'dev', f'{subject_name}_dev.csv')
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.csv"
        )
        # val_df = pd.read_csv(val_file_path, names=['question','A','B','C','D','answer'])
        # dev_df = pd.read_csv(dev_file_path, names=['question','A','B','C','D','answer'])
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        ).astype(str)

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            test_df,
            save_result_dir = args.save_result_dir,
            overwrite=args.overwrite,
        )
        dev_result[subject_name] = score
    cal_mmlu(dev_result)


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
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
# 等价于下面的代码
# SUBJECTS = []
# for vl in TASK_NAME_MAPPING.values():  # 遍历每个类别的任务列表
#     for v in vl:  # 遍历当前类别的每个任务名称
#         SUBJECTS.append(v)  # 将任务名称添加到 SUBJECTS 列表中
choices = ["A", "B", "C", "D"]

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="/data3/whr/wyl/Llama-3-8B-Instruct",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--save_result_dir",
        type=str,
        help="Path to save the result",
        
    )
    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, help="Path to eval data")
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existed results",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
    end = time.time()
    print(f"Time cost: {end-start} s")