from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#设置原来本地模型的地址
model_name_or_path = ''
#设置微调后模型的地址
adapter_name_or_path = ''
#设置合并后模型的导出地址
save_path = ''

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)
# tokenizer.padding_side = 'left'
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,

    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
print("load model success")
# with torch.no_grad():
#     model.resize_token_embeddings(len(tokenizer))
#     model.config.pad_token_id = tokenizer.pad_token_id 
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter success")
model = model.merge_and_unload()
print("merge success")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("save done.")
