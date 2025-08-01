"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask for assistant responses. Supports both Qwen and Llama formats.
    NOTE: This assumes that the input_ids contains alternating user and assistant turns
    """
    # Detect model type and set appropriate tokens
    model_name = getattr(tokenizer, 'name_or_path', '').lower()
    
    if 'qwen' in model_name:
        # Qwen format: <|im_start|>assistant ... <|im_end|>
        try:
            special_token = tokenizer.encode("<|im_start|>")[0]
            reward_token = tokenizer.encode("<|im_end|>")[0]
            use_qwen_format = True
        except:
            use_qwen_format = False
    else:
        use_qwen_format = False
    
    if use_qwen_format:
        # Original Qwen logic
        turn_starts = torch.where(input_ids == special_token, 1, 0)
        turn_indicators = torch.cumsum(turn_starts, dim=-1)
        response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    else:
        # Llama format: properly identify assistant response sections
        batch_size, seq_len = input_ids.shape
        
        # Initialize masks as all False
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Get Llama special tokens
        try:
            start_header_id = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)[0]
            end_header_id = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)[0]
            eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
            assistant_token = tokenizer.encode("assistant", add_special_tokens=False)[0]
        except Exception as e:
            print(f"[WARNING] Could not get Llama special tokens: {e}")
            # Fallback to conservative approach if tokens not found
            loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
            response_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            # For each batch item, find assistant response sections
            for batch_idx in range(batch_size):
                sequence = input_ids[batch_idx]
                
                # Find all header starts
                header_starts = torch.where(sequence == start_header_id)[0]
                
                for header_start in header_starts:
                    # Check if this is an assistant header
                    if (header_start + 1 < seq_len and 
                        sequence[header_start + 1] == assistant_token and
                        header_start + 2 < seq_len and 
                        sequence[header_start + 2] == end_header_id):
                        
                        # Found assistant header, mark content as valid
                        content_start = header_start + 3  # After <|start_header_id|>assistant<|end_header_id|>
                        
                        # Find the corresponding <|eot_id|>
                        eot_positions = torch.where(sequence[content_start:] == eot_id)[0]
                        if len(eot_positions) > 0:
                            content_end = content_start + eot_positions[0]  # Exclusive end
                            # Mark assistant content as valid for learning
                            loss_mask[batch_idx, content_start:content_end] = True
                            response_mask[batch_idx, content_start:content_end] = True
        
        # Handle padding tokens
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            pad_mask = (input_ids != tokenizer.pad_token_id)
            loss_mask = loss_mask & pad_mask
            response_mask = response_mask & pad_mask

    # Handle rewards
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores and use_qwen_format:
        # Only use turn-based scoring for Qwen format
        for idx, scores in enumerate(list(zip(*all_scores))):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            score_tensor[reward_position] = scores
    else:
        # Place rewards at the end of sequences
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    
    # Remove last token from masks and first token from scores (standard RLHF practice)
    loss_mask = loss_mask[:, :-1]
    score_tensor = score_tensor[:, 1:]
    response_mask = response_mask[:, :-1]

    # Final safety check: ensure masks are not all zeros
    if loss_mask.sum() == 0:
        print(f"[WARNING] loss_mask is all zeros in get_masks_and_scores! Forcing the last token to be valid.")
        loss_mask[:, -1] = True
    if response_mask.sum() == 0:
        print(f"[WARNING] response_mask is all zeros in get_masks_and_scores! Forcing the last token to be valid.")
        response_mask[:, -1] = True

    return loss_mask, score_tensor, response_mask



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
        # Fix Llama tokenizer's pad_token issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[ContextManager] Setting pad_token to eos_token: {self.tokenizer.pad_token}")

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group

        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)


            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions

    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"

        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}


        # apply penalty pre-normalization
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output["penalty"] for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor

    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        llm_input_texts = []
        messages_list = [] # for api calling
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            messages = [
                {"role": "system", "content": f"You're a helpful assistant. "}, 
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]

            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                    messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})


            # NOTE: this assertion is important for loss mask computation        
            assert all(msg["role"] == "assistant" for msg in messages[2::2])

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
            if not prepare_for_update:
                if self.config.agent_proxy.enable_think:
                    text += "<think>" # force the LLM to think before answering
                else:
                    text += "<answer>" # force the LLM to answer
            llm_input_texts.append(text)
            messages_list.append(messages)

        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        if prepare_for_update:
            scores = [[i['reward'] for i in env_output['history']] for env_output in env_outputs]
            loss_mask, score_tensor, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores, use_turn_scores=self.config.agent_proxy.use_turn_scores)
            if self.config.enable_response_mask:
                loss_mask = response_mask
            normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor # remove the first token
            llm_inputs.batch["response_mask"] = response_mask # add response_mask for training
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }

            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for response in responses] # The LLM generation does not include <think> tags. Add them back here.

        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs





@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print("ctx_manager prefix", ctx_manager.prefix_lookup)
    batch_list = [
        {
            "env_id": 0,
            "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
        },
        {
            "env_id": 1,
            "chat_response": "<think> 456. </think><answer> love ; you </answer><think> mlll nb </think><answer> lxxx ; you </answer>",
        }
    ]
    ctx_manager.action_sep_lookup = {
        0: "|",
        1: ";"
    }
    for item in batch_list:
        item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    batch_dict = collate_fn(batch_list)
    batch = DataProto.from_single_dict(batch_dict)
    env_inputs = ctx_manager.get_env_inputs(batch)
    print(env_inputs)



    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8},
                {"state": "###\n#x_#<image>"}
            ],
            "group_id": 0
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3},
                {"state": "###\n#x_#<image>"}
            ],
            "group_id": 1
        }
    ]

    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)

if __name__ == "__main__":
    main()
