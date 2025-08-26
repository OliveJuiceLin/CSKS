from typing import Optional, Dict, Any
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
#import accelerate
#device = "balanced_low_0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep),
                    logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def compute_outputs(model, inputs, return_dict):
    with torch.no_grad():
        return model(**inputs, return_dict=return_dict)

class DExpertsLlama:
    def __init__(
        self,
        base,
        expert,
        antiexpert,
        tokenizer: PreTrainedTokenizer,
        # system_prompt: str = None,
        #alpha: float = 1.0,
        # chat_response_prefix: str = None,
    ):

        self.tokenizer = tokenizer
        self.base = base
        self.expert = expert
        self.antiexpert = antiexpert


        #self.alpha = alpha


    def paralell_forward(self,
                         base_inputs,
                         expert_inputs,
                         antiexpert_inputs,
                         return_dict=None):
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_base = executor.submit(compute_outputs, self.base, base_inputs, return_dict)
            future_expert = executor.submit(compute_outputs, self.expert, expert_inputs, return_dict)
            future_antiexpert = executor.submit(compute_outputs, self.antiexpert, antiexpert_inputs, return_dict)

            # 获取结果
            base_outputs = future_base.result()
            expert_outputs = future_expert.result()
            antiexpert_outputs = future_antiexpert.result()

        return base_outputs, expert_outputs, antiexpert_outputs
    def paralell_forward_without_antiexpert(self,
                         base_inputs,
                         expert_inputs,
                         
                         return_dict=None):
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_base = executor.submit(compute_outputs, self.base, base_inputs, return_dict)
            future_expert = executor.submit(compute_outputs, self.expert, expert_inputs, return_dict)
            
            # 获取结果
            base_outputs = future_base.result()
            expert_outputs = future_expert.result()
            
        return base_outputs, expert_outputs
    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs,
        return_dict=None
    ):
        with torch.no_grad():
            base_outputs = self.base(**base_inputs, return_dict=return_dict)
            expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
            antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)
        
        
        
        
        return base_outputs, expert_outputs, antiexpert_outputs

    def proxy_forward_once(
        self,
        unprepared_inputs,
        return_dict=True,
        alpha = 0):
        
        base_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        expert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        antiexpert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        base_prepared_inputs =          self.base.prepare_inputs_for_generation(**base_unprepared_inputs)
        expert_prepared_inputs = self.expert.prepare_inputs_for_generation(
            **expert_unprepared_inputs)
        antiexpert_prepared_inputs = self.antiexpert.prepare_inputs_for_generation(
            **antiexpert_unprepared_inputs)
        with torch.no_grad():
            base_outputs = self.base(
                **base_prepared_inputs, return_dict=return_dict)
            expert_outputs = self.expert(
                **expert_prepared_inputs, return_dict=return_dict)
            antiexpert_outputs = self.antiexpert(
                **antiexpert_prepared_inputs, return_dict=return_dict)
            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
            next_token_logits = (base_next_token_logits + alpha *(expert_next_token_logits - antiexpert_next_token_logits))
            logits_diff = expert_next_token_logits - antiexpert_next_token_logits
            # if (input("是否查看差异? y/n: ") == 'y'):
            #     return next_token_logits,logits_diff
            return next_token_logits,logits_diff,base_next_token_logits
        
    def proxy_forward_once_without_antiexpert(
        self,
        unprepared_inputs,
        return_dict=True,
        alpha = 0):
        
        base_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        expert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        #antiexpert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        base_prepared_inputs =          self.base.prepare_inputs_for_generation(**base_unprepared_inputs)
        expert_prepared_inputs = self.expert.prepare_inputs_for_generation(
            **expert_unprepared_inputs)
        #antiexpert_prepared_inputs = self.antiexpert.prepare_inputs_for_generation(**antiexpert_unprepared_inputs)
        with torch.no_grad():
            base_outputs = self.base(
                **base_prepared_inputs, return_dict=return_dict)
            expert_outputs = self.expert(
                **expert_prepared_inputs, return_dict=return_dict)
            #antiexpert_outputs = self.antiexpert(**antiexpert_prepared_inputs, return_dict=return_dict)
            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            #antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
            next_token_logits = (base_next_token_logits + alpha *(expert_next_token_logits - base_next_token_logits))
            logits_diff = expert_next_token_logits - base_next_token_logits
            # if (input("是否查看差异? y/n: ") == 'y'):
            #     return next_token_logits,logits_diff
            return next_token_logits,logits_diff,base_next_token_logits
            
        
        
    def generate(
        self,
        unprepared_inputs,
        max_new_tokens: Optional[int] = 100,
        anti_expert:bool = True,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        alpha: float = 1.0,
        fire:float = 0.0,
        #**kwargs
    ):

        base_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        expert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        antiexpert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        
        
        unfinished_sequences = torch.ones(
            unprepared_inputs['input_ids'].shape[0], dtype=torch.long, device=unprepared_inputs['input_ids'].device)
        eos_token_id_tensor = torch.tensor(
            [self.tokenizer.eos_token_id]).to(unprepared_inputs['input_ids'].device)

        #if return_logits_for_analysis:
        #    analysis_data = defaultdict(list)
        with torch.no_grad():
            for step in range(max_new_tokens):
                
                
                # prepare model inputs with past_key_values and attention_mask
                base_inputs = self.base.prepare_inputs_for_generation(**base_unprepared_inputs)
                expert_inputs = self.expert.prepare_inputs_for_generation(
                    **expert_unprepared_inputs)
                antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(
                    **antiexpert_unprepared_inputs)

                # DExperts
                base_outputs, expert_outputs, antiexpert_outputs = self.paralell_forward(
                    base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
                )

                base_next_token_logits = base_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_outputs.logits[..., -1, :]
                antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
                # DExperts!
                if (step != 0 or (step == 0 and fire == 0.0)):
                    next_token_logits = ( base_next_token_logits + alpha * (expert_next_token_logits - antiexpert_next_token_logits) )
                elif (fire != 0.0 and step == 0):
                    next_token_logits = ( base_next_token_logits + fire * (expert_next_token_logits - antiexpert_next_token_logits) )
                    
               
                    

               

                # # warp logits
                # if temperature != 1.0:
                #     next_token_logits = next_token_logits / temperature
                # if top_p < 1.0:
                #     next_token_logits = top_k_top_p_filtering(
                #         next_token_logits, top_p=top_p, top_k=10)

                # decode
                # 是否开启采样，默认是 False，即贪婪找最大条件概率的词。
                if do_sample:
                    # warp logits
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    if top_p < 1.0:
                        next_token_logits = top_k_top_p_filtering(
                            next_token_logits, top_p=top_p, top_k=10)
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(
                        probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                
                
                # 查看过程
                # print(f"next_tokens: {next_tokens}")
                # print(
                #     f"next_token_text: {repr(self.tokenizer.decode(next_tokens))}")
                # print(f"="*100)
                next_tokens = (
                    next_tokens * unfinished_sequences +
                    self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                )
                
                # update model inputs for next step
                base_unprepared_inputs['input_ids'] = torch.cat([base_unprepared_inputs['input_ids'], next_tokens[:, None]], dim=-1)
                expert_unprepared_inputs['input_ids'] = torch.cat([expert_unprepared_inputs['input_ids'], next_tokens[:, None]], dim=-1)
                antiexpert_unprepared_inputs['input_ids'] = torch.cat([antiexpert_unprepared_inputs['input_ids'], next_tokens[:, None]], dim=-1)
                
                # update kwargs
                base_unprepared_inputs = self._update_model_kwargs_for_generation(
                    base_outputs, base_unprepared_inputs)
                expert_unprepared_inputs = self._update_model_kwargs_for_generation(
                    expert_outputs, expert_unprepared_inputs)
                antiexpert_unprepared_inputs = self._update_model_kwargs_for_generation(
                    antiexpert_outputs, antiexpert_unprepared_inputs)
                

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(
                        eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    break
            #torch.cuda.empty_cache()

        return base_unprepared_inputs['input_ids']
    def generate_without_antiexpert(
        self,
        unprepared_inputs,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        alpha: float = 1.0,
        fire:float = 0.0,
        #**kwargs
    ):

        base_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        expert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        #antiexpert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        
        
        unfinished_sequences = torch.ones(
            unprepared_inputs['input_ids'].shape[0], dtype=torch.long, device=unprepared_inputs['input_ids'].device)
        eos_token_id_tensor = torch.tensor(
            [self.tokenizer.eos_token_id]).to(unprepared_inputs['input_ids'].device)

        #if return_logits_for_analysis:
        #    analysis_data = defaultdict(list)
        with torch.no_grad():
            for step in range(max_new_tokens):
                
                
                # prepare model inputs with past_key_values and attention_mask
                base_inputs = self.base.prepare_inputs_for_generation(**base_unprepared_inputs)
                expert_inputs = self.expert.prepare_inputs_for_generation(**expert_unprepared_inputs)

                # DExperts
                base_outputs, expert_outputs = self.paralell_forward_without_antiexpert(
                    base_inputs, expert_inputs,return_dict=True
                )

                base_next_token_logits = base_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_outputs.logits[..., -1, :]
                
                # DExperts!
                # Fire!
                if (step != 0 or (step == 0 and fire == 0.0)):
                    next_token_logits = ( base_next_token_logits + alpha * (expert_next_token_logits) )
                elif (fire != 0.0 and step == 0):
                    next_token_logits = ( base_next_token_logits + fire * (expert_next_token_logits) )

                # warp logits
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_p < 1.0:
                    next_token_logits = top_k_top_p_filtering(
                        next_token_logits, top_p=top_p, top_k=10)

                # decode
                # 是否开启采样，默认是 False，即贪婪找最大条件概率的词。
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(
                        probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                
                
                # 查看过程
                # print(f"next_tokens: {next_tokens}")
                # print(
                #     f"next_token_text: {repr(self.tokenizer.decode(next_tokens))}")
                # print(f"="*100)
                next_tokens = (
                    next_tokens * unfinished_sequences +
                    self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                )
                
                # update model inputs for next step
                base_unprepared_inputs['input_ids'] = torch.cat([base_unprepared_inputs['input_ids'], next_tokens[:, None]], dim=-1)
                expert_unprepared_inputs['input_ids'] = torch.cat([expert_unprepared_inputs['input_ids'], next_tokens[:, None]], dim=-1)
                
                # update kwargs
                base_unprepared_inputs = self._update_model_kwargs_for_generation(
                    base_outputs, base_unprepared_inputs)
                expert_unprepared_inputs = self._update_model_kwargs_for_generation(
                    expert_outputs, expert_unprepared_inputs)
               
                

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(
                        eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    break
            #torch.cuda.empty_cache()

        return base_unprepared_inputs['input_ids']
    def generate_with_context_aware_decoding(
        self,
        unprepared_inputs,
        unprepared_inputs_without_context,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        alpha: float = 1.0,
        fire:float = 0.0,
        #**kwargs
    ):

        base_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        expert_unprepared_inputs = copy.deepcopy(unprepared_inputs)
        antiexpert_unprepared_inputs = copy.deepcopy(unprepared_inputs_without_context)
        
        
        unfinished_sequences = torch.ones(
            unprepared_inputs['input_ids'].shape[0], dtype=torch.long, device=unprepared_inputs['input_ids'].device)
        eos_token_id_tensor = torch.tensor(
            [self.tokenizer.eos_token_id]).to(unprepared_inputs['input_ids'].device)

        #if return_logits_for_analysis:
        #    analysis_data = defaultdict(list)
        with torch.no_grad():
            for step in range(max_new_tokens):
                
                
                # prepare model inputs with past_key_values and attention_mask
                base_inputs = self.base.prepare_inputs_for_generation(**base_unprepared_inputs)
                expert_inputs = self.expert.prepare_inputs_for_generation(
                    **expert_unprepared_inputs)
                antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(
                    **antiexpert_unprepared_inputs)

                # DExperts
                base_outputs, expert_outputs, antiexpert_outputs = self.paralell_forward(
                    base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
                )

                base_next_token_logits = base_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_outputs.logits[..., -1, :]
                antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
                # DExperts!
                if (step != 0 or (step == 0 and fire == 0.0)):
                    next_token_logits = ( base_next_token_logits + alpha * (expert_next_token_logits - antiexpert_next_token_logits) )
                elif (fire != 0.0 and step == 0):
                    next_token_logits = ( base_next_token_logits + fire * (expert_next_token_logits - antiexpert_next_token_logits) )
                    
               
                    

               

                # warp logits
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_p < 1.0:
                    next_token_logits = top_k_top_p_filtering(
                        next_token_logits, top_p=top_p, top_k=10)

                # decode
                # 是否开启采样，默认是 False，即贪婪找最大条件概率的词。
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(
                        probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                
                
                # 查看过程
                # print(f"next_tokens: {next_tokens}")
                # print(
                #     f"next_token_text: {repr(self.tokenizer.decode(next_tokens))}")
                # print(f"="*100)
                next_tokens = (
                    next_tokens * unfinished_sequences +
                    self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                )
                
                # update model inputs for next step
                base_unprepared_inputs['input_ids'] = torch.cat([base_unprepared_inputs['input_ids'], next_tokens[:, None]], dim=-1)
                expert_unprepared_inputs['input_ids'] = torch.cat([expert_unprepared_inputs['input_ids'], next_tokens[:, None]], dim=-1)
                antiexpert_unprepared_inputs['input_ids'] = torch.cat([antiexpert_unprepared_inputs['input_ids'], next_tokens[:, None]], dim=-1)
                
                # update kwargs
                base_unprepared_inputs = self._update_model_kwargs_for_generation(
                    base_outputs, base_unprepared_inputs)
                expert_unprepared_inputs = self._update_model_kwargs_for_generation(
                    expert_outputs, expert_unprepared_inputs)
                antiexpert_unprepared_inputs = self._update_model_kwargs_for_generation(
                    antiexpert_outputs, antiexpert_unprepared_inputs)
                

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(
                        eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    break
            #torch.cuda.empty_cache()

        return base_unprepared_inputs['input_ids']

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        inputs,
    ):
        # update past_key_values
        inputs["past_key_values"] = outputs.past_key_values

        # update attention mask

        attention_mask = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
    )
        
        # update cache_position
        if "cache_position" in inputs:
            inputs["cache_position"] = inputs["cache_position"] + 1
        else:
        # Initialize cache_position if it doesn't exist
            inputs["cache_position"] = torch.tensor([inputs["attention_mask"].shape[1] - 1], device=attention_mask.device)
        # position_ids
        seq_len_with_past = inputs['input_ids'].shape[1]
        past_seq_len = inputs['past_key_values'][0][0].shape[-2] if inputs['past_key_values'] else 0
        new_seq_len = seq_len_with_past - past_seq_len
        position_ids = torch.arange(past_seq_len, new_seq_len + past_seq_len,
                                dtype=torch.long, device=inputs['input_ids'].device)
        position_ids = position_ids.unsqueeze(0).view(-1, new_seq_len)
        inputs['position_ids'] = position_ids

        return inputs



