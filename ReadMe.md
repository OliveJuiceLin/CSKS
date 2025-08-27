# CSKS
This repo contains code for the main experiments in our paper:

**Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models**

## Repo Structure
```bash
├─CONSTRUCT_DATA
│  ├─MUSIQUE
│  │  ├─TEST_CODE
│  │  │  ├─blackBox
│  │  │  ├─LLAMA
│  │  │  └─Qwen
│  │  └─TEST_RESULT
│  │      ├─blackBox
│  │      │  ├─llama
│  │      │  └─qwen
│  │      ├─gemma
│  │      ├─llama
│  │      └─qwen
│  └─POP_QA
│      ├─TEST_CODE
│      │  ├─blackBox
│      │  ├─LLAMA
│      │  └─Qwen
│      └─TEST_RESULT
│          ├─blackBox
│          ├─gemma
│          ├─llama
│          └─qwen
├─eval-mmlu
│  ├─data
│  └─outs
├─FINE-TUNING-CONTEXT
│  └─data
├─FINE-TUNING-PARAMETRIC
│  ├─data_long_most_similar
│  └─data_short_least_similar
└─proxy_model
    └─__pycache__
```
## proxy_model
- This folder defines the CSKS inference framework that corporates the target model and the proxy models, corresponding to **Section 2.1 CSKS Framework** in our paper.

## FINE-TUNING-*
- This folder contains the code and data used for tunning our proxy models, corresponding to **Appendix A Finetune Dataset Construction Details**

## CONSTRUCT_DATA

- This folder contains two subfolder, MUSIQUE and POP_QA, representing the original dataset which we will transform later.

- Under each subfolder, we provides the pipeline that transforms the original dataset to a more refined knowledge conflict evaluation dataset corresponding to **Section 2.2 Evaluation Method** in our paper.

- Besides we also provides the code (TEST_CODE folder) that evaluates CSKS and other baseline methods on our refined dataset, corresponding to **Section 3.3 Results** and **Section 3.5 Analysis**