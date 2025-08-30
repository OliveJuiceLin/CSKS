# CSKS: Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models
---
This repository contains the official implementation of our paper:

**Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models**

Paper link: https://arxiv.org/abs/2508.19720
## Abstract

In Large Language Models (LLMs) generation, there exist knowledge conflicts where parametric knowledge contradicts knowledge provided in the context. We propose **CSKS** (Continuously Steering Knowledge Sensitivity), a simple framework that can steer LLMs' sensitivity to contextual knowledge continuously at a lightweight cost. Specifically, we tune two small LMs (proxy models) and use the difference in their output distributions to shift the original distribution of an LLM without modifying the LLM weights.

## 🔥 Key Features

- **Continuous Control**: Precisely adjust LLMs' sensitivity to contextual knowledge via a single hyperparameter α
- **Lightweight**: Uses small proxy models (~7B) to steer large models (~70B) without modifying target model weights
- **Bidirectional**: Both increase and decrease sensitivity to contextual knowledge (α > 0 for context-faithful, α < 0 for parametric-faithful)
- **Black-box Compatible**: Works with API-based models like GPT-3.5-Turbo
- **Model Agnostic**: Supports different model families (LLaMA, Qwen, Gemma)

## 🚀 Quick Start

### Environment Setup

```bash
conda create -n csks python=3.9
conda activate csks
pip install -r requirements.txt
```

### Basic Usage

```python
from proxy_model.dexpert import DExpertsLlama
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load models
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
expert_model = AutoModelForCausalLM.from_pretrained("path/to/context-faithful-model")
antiexpert_model = AutoModelForCausalLM.from_pretrained("path/to/parametric-faithful-model")

# Initialize CSKS framework
csks_model = DExpertsLlama(
    base=base_model,
    expert=expert_model,
    antiexpert=antiexpert_model,
    tokenizer=tokenizer
)

# Generate with controlled sensitivity
inputs = tokenizer("Your input text", return_tensors="pt")
output = csks_model.generate(
    inputs,
    alpha=1.0,  # Positive for context-faithful, negative for parametric-faithful
    max_new_tokens=100
)
```

## 📊 Evaluation

### Dataset Construction

We provide scripts to construct evaluation datasets with controlled knowledge conflicts:

```bash
# For MuSiQue dataset
cd CONSTRUCT_DATA/MUSIQUE
python 1_GET_QA.py
python 2_FLITER.py
python 3_PROCESS-triplet.py
python 4-REMOVE_REPITION.py
python 5-distractAndcontext.py

# For PopQA dataset
cd CONSTRUCT_DATA/POP_QA
python popqa-1-get-qa.py
python popqa-2-fliter.py
python popqa-3-get-popularity.py
python popqa-4-distract.py
python popqa-5-score.py
```

### Running Experiments


```bash
# LLaMA (Qwen, gemma) models on MuSiQue (PopQA)
cd CONSTRUCT_DATA/MUSIQUE/TEST_CODE/LLAMA
python TEST_ON_MUSIQUE_PROXY_MODEL.py \
    --base_model_name "path_to_base_model" \
    --expert_model_name "path_to_expert_model" \
    --antiexpert_model_name "path_to_anti-expert_model" \
    --tokenizer_path "path_to_tokenizer" \
    --data_path "path_to_your_input.json" \
    --output_path "path_to_your_output.json" \
    --alpha -0.5 \
    --cuda_devices "0,1,3"

# Qwen models on PopQA
cd CONSTRUCT_DATA/POP_QA/TEST_CODE/Qwen
python POP_QA_QWEN_PROXY.py --alpha 1.0 --dataset_path path/to/dataset
```

### Evaluation on MMLU
```bash
cd eval-mmlu
python mmlu.py --model_path path/to/model --alpha 1.0
```

## 🔧 Training Proxy Models

### Context-Faithful Model (Expert)
```bash
cd FINE-TUNING-CONTEXT
python FINE-TUING.py \
    --model_id "your_model_id" \
    --output_dir "your_output_directory" \
    --train_data_path "path/to/train.csv" \
    --val_data_path "path/to/validation.csv" \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lora_r 16 \
    --lora_alpha 32
```

### Parametric-Faithful Model (Anti-Expert)
```bash
cd FINE-TUNING-PARAMETRIC
python FINE-TUNING.py \
    --model_id "your_base_model_id" \
    --output_dir "path/to/your/output" \
    --train_data_path "path/to/train.json" \
    --val_data_path "path/to/val.json" \
    --cuda_devices "2" \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2
```

## 📁 Repository Structure

```
├── proxy_model/                    # Core CSKS framework implementation
│   └── dexpert.py                 # Main DExpertsLlama class
├── CONSTRUCT_DATA/                # Dataset construction and evaluation
│   ├── MUSIQUE/                   # MuSiQue dataset processing
│   │   ├── TEST_CODE/            # Evaluation scripts
│   │   └── TEST_RESULT/          # Experimental results
│   └── POP_QA/                   # PopQA dataset processing
│       ├── TEST_CODE/            # Evaluation scripts
│       └── TEST_RESULT/          # Experimental results
├── FINE-TUNING-CONTEXT/          # Context-faithful model training
├── FINE-TUNING-PARAMETRIC/       # Parametric-faithful model training
└── eval-mmlu/                    # MMLU evaluation scripts
```

## 📈 Results

Our method achieves significant improvements in sensitivity scores:


#### MusiQue ‧ LLaMA-3-Instruct
| Method | Degree-1 | Degree-2 | Context-1 | Context-2 | Pop-1 | Pop-2 | Pop-3 | Sensitivity |
|--------|----------|----------|-----------|-----------|-------|-------|-------|-------------|
| Origin | 64.85 | 20.17 | 55.08 | 30.00 | 49.44 | 42.63 | 35.71 | 38.13 |
| **CSKS** | **78.08** (+13.23) | **60.38** (+40.21) | **79.97** (+24.89) | **58.53** (+28.53) | **75.27** (+25.83) | **65.84** (+23.21) | **66.66** (+30.95) | **66.72** (+28.59) |

#### MusiQue ‧ Qwen2.5-Instruct
| Method | Degree-1 | Degree-2 | Context-1 | Context-2 | Pop-1 | Pop-2 | Pop-3 | Sensitivity |
|--------|----------|----------|-----------|-----------|-------|-------|-------|-------------|
| Origin | 69.85 | 23.71 | 57.29 | 36.32 | 53.00 | 47.54 | 40.04 | 42.58 |
| **CSKS** | **94.85** (+25.00) | **85.13** (+61.42) | **90.43** (+33.14) | **89.56** (+53.24) | **93.54** (+40.54) | **85.94** (+38.40) | **90.47** (+50.43) | **89.26** (+46.68) |

#### PopQA ‧ LLaMA-3-Instruct
| Method | Degree-1 | Degree-2 | Context-1 | Context-2 | Pop-1 | Pop-2 | Pop-3 | Sensitivity |
|--------|----------|----------|-----------|-----------|-------|-------|-------|-------------|
| Origin | 52.04 | 23.62 | 52.21 | 23.48 | 43.14 | 37.29 | 33.22 | 34.32 |
| **CSKS** | **69.79** (+17.75) | **65.45** (+41.83) | **80.46** (+28.25) | **54.80** (+31.32) | **66.72** (+23.58) | **67.72** (+30.43) | **68.40** (+35.18) | **66.24** (+31.92) |

#### PopQA ‧ Qwen2.5-Instruct
| Method | Degree-1 | Degree-2 | Context-1 | Context-2 | Pop-1 | Pop-2 | Pop-3 | Sensitivity |
|--------|----------|----------|-----------|-----------|-------|-------|-------|-------------|
| Origin | 66.15 | 28.59 | 60.60 | 34.18 | 51.67 | 47.83 | 42.79 | 43.59 |
| **CSKS** | **93.83** (+27.68) | **90.40** (+61.81) | **93.27** (+32.67) | **90.96** (+56.78) | **88.46** (+36.79) | **93.14** (+45.31) | **94.65** (+51.86) | **92.24** (+48.65) |


## 🎛️ Hyperparameter Control

The α parameter provides continuous control over knowledge sensitivity:

- **α > 0**: Increases sensitivity to contextual knowledge
- **α = 0**: Original model behavior
- **α < 0**: Increases reliance on parametric knowledge

## Black-box Model Support

For API-based models, use the limited logits version:

```bash
python blackBox.py \
    --api_key "YOUR_OPENAI_API_KEY" \
    --base_url "YOUR_API_BASE_URL" \
    --black_box_model "gpt-3.5-turbo" \
    --expert_model_name "path_to_expert_model" \
    --antiexpert_model_name "path_to_anti-expert_model" \
    --tokenizer_path "path_to_tokenizer" \
    --data_path "path/to/your/data.json" \
    --output_path_origin "path/to/origin_results.json" \
    --output_path_proxy "path/to/proxy_results.json" \
    --alpha 0.7 \
    --cuda_devices "0"
```

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2025continuously,
  title={Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models},
  author={Wang, Yilin and Wang, Heng and Bai, Yuyang and Luo, Minnan},
  journal={arXiv preprint arXiv:2508.19720},
  year={2025}
}
```





## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [13148035071xjtu@stu.xjtu.edu.cn](mailto:13148035071xjtu@stu.xjtu.edu.cn)

---

**Note**: This repository is under active development.

