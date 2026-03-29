
from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .base_llm import ConcurrentLLM
import time

class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name,
            enable_sleep_mode=True,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
		)
		print("LLM initialized")
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs
	
class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs
        
        
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses, 
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
		self.train_es_manager = EnvStateManager(config, mode="train")
		self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer

	def generate_sequences(self, lm_inputs: DataProto):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, RayWorkerGroup):
			padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
			padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
			lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
			lm_outputs.meta_info = lm_inputs.meta_info
			lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
		elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(self.actor_wg, ApiCallingWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs

	def rollout(self, dataproto: DataProto, val=False, seed=None):
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		env_outputs = es_manager.reset(seed=seed)
		max_turns = self.config.val_agent_proxy.max_turn if val else self.config.agent_proxy.max_turn
		use_rollout_session = isinstance(self.actor_wg, RayWorkerGroup) and hasattr(self.actor_wg, "start_rollout_session")
		show_eval_progress = bool(val and self.config.trainer.get('eval_show_progress', False))
		progress_interval = int(self.config.trainer.get('eval_progress_interval', 1) or 1)
		progress_interval = max(progress_interval, 1)
		progress_start_time = time.time() if show_eval_progress else None
		total_by_tag = {}
		completed_tags = set()
		last_printed_turn_idx = -1

		# Track which turn each environment succeeds in (0-indexed, -1 means never succeeded)
		num_envs = len(es_manager.envs)
		success_at_turn = [-1] * num_envs
		last_turn_idx = -1

		if show_eval_progress:
			for entry in es_manager.envs:
				tag = entry['tag']
				total_by_tag[tag] = total_by_tag.get(tag, 0) + 1
			env_breakdown = ", ".join(f"{tag}={count}" for tag, count in total_by_tag.items())
			print(
				f"[EVAL] Validation rollout started | total_envs={num_envs} | "
				f"max_turns={max_turns} | envs: {env_breakdown}"
			)

		def _format_elapsed(seconds):
			total_seconds = max(int(seconds), 0)
			hours = total_seconds // 3600
			minutes = (total_seconds % 3600) // 60
			secs = total_seconds % 60
			return f"{hours:02d}:{minutes:02d}:{secs:02d}"

		def _print_eval_progress(turn_idx, newly_done=0, force=False):
			nonlocal last_printed_turn_idx

			if not show_eval_progress:
				return
			turn_number = turn_idx + 1
			if turn_idx == last_printed_turn_idx:
				return
			if (not force) and (turn_number % progress_interval != 0):
				return

			done_by_tag = {}
			success_by_tag = {}
			done_total = 0
			success_total = 0
			for entry in es_manager.envs:
				tag = entry['tag']
				status = entry['status']
				done = int(status.terminated)
				success = int(status.terminated and not status.truncated)
				done_by_tag[tag] = done_by_tag.get(tag, 0) + done
				success_by_tag[tag] = success_by_tag.get(tag, 0) + success
				done_total += done
				success_total += success

			elapsed = time.time() - progress_start_time if progress_start_time is not None else 0.0
			elapsed_hms = _format_elapsed(elapsed)
			env_parts = []
			for tag, total in total_by_tag.items():
				done = done_by_tag.get(tag, 0)
				success = success_by_tag.get(tag, 0)
				active = total - done
				env_parts.append(f"{tag}: done={done}/{total}, succ={success}, active={active}")
				if done == total and tag not in completed_tags:
					completed_tags.add(tag)
					print(
						f"[EVAL] Environment completed | tag={tag} | "
						f"done={done}/{total} | elapsed={elapsed:.1f}s ({elapsed_hms})"
					)

			done_pct = (done_total / num_envs) if num_envs > 0 else 0.0
			print(
				f"[EVAL] Turn {turn_number}/{max_turns} | done={done_total}/{num_envs} ({done_pct:.1%}) | "
				f"succ={success_total} | newly_done={newly_done} | elapsed={elapsed:.1f}s ({elapsed_hms}) | "
				+ " ; ".join(env_parts)
			)
			last_printed_turn_idx = turn_idx

		if use_rollout_session:
			self.actor_wg.start_rollout_session()

		try:
			for i in range(max_turns):
				last_turn_idx = i
				done_before = sum(int(entry['status'].terminated) for entry in es_manager.envs)
				lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
				lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done
				lm_outputs: DataProto = self.generate_sequences(lm_inputs)
				env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
				env_outputs: List[Dict] = es_manager.step(env_inputs)

				# Track success at this turn (using existing success logic)
				for env_entry in es_manager.envs:
					env_id = env_entry['env_id']
					status = env_entry['status']
					# Success is defined as: terminated and not truncated
					if success_at_turn[env_id] == -1 and status.terminated and not status.truncated:
						success_at_turn[env_id] = i

				done_after = sum(int(entry['status'].terminated) for entry in es_manager.envs)
				_print_eval_progress(i, newly_done=done_after - done_before)

				if len(env_outputs) == 0: # all finished
					break
		finally:
			if use_rollout_session:
				self.actor_wg.end_rollout_session()

		if show_eval_progress:
			final_turn_idx = last_turn_idx if last_turn_idx >= 0 else 0
			_print_eval_progress(final_turn_idx, newly_done=0, force=True)

		rollout_states = es_manager.get_rollout_states()
		rollouts = ctx_manager.formulate_rollouts(rollout_states)

		# Compute pass@k metrics if in validation mode
		if val:
			episode_records = es_manager.get_episode_records()
			for record in episode_records:
				record['success_at_turn'] = success_at_turn[record['env_id']]
			pass_at_k_metrics = self._compute_pass_at_k(success_at_turn, max_turns)
			rollouts.meta_info['pass_at_k'] = pass_at_k_metrics
			rollouts.meta_info['success_at_turn'] = success_at_turn
			rollouts.meta_info['episode_records'] = episode_records

		# self.tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=False) # see all the trajectories
		return rollouts

	def _compute_pass_at_k(self, success_at_turn, max_turns):
		"""Compute pass@k metrics from success_at_turn data"""
		import numpy as np
		success_at_turn = np.array(success_at_turn)
		total = len(success_at_turn)

		pass_at_k = {}
		for k in range(1, max_turns + 1):
			# Count how many succeeded within first k turns
			# success_at_turn[i] = turn index (0-indexed) when env i succeeded, or -1 if never succeeded
			# We want to count envs where 0 <= success_at_turn < k
			succeeded = np.sum((success_at_turn >= 0) & (success_at_turn < k))
			pass_at_k[f'pass@{k}'] = succeeded / total if total > 0 else 0.0

		return pass_at_k

@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(config):
	os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = VllmWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	import time
	start_time = time.time()
	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)
	end_time = time.time()
	print(f'rollout time: {end_time - start_time} seconds')
	# print rollout rewards from the rm_scores
	rm_scores = rollouts.batch["rm_scores"]
	metrics = rollouts.meta_info["metrics"]
	avg_reward = rm_scores.sum(-1).mean().item()
	print(f'rollout rewards: {avg_reward}')
	print(f'metrics:')
	for k, v in metrics.items():
		print(f'{k}: {v}')

	# Print pass@k metrics if available
	if 'pass_at_k' in rollouts.meta_info:
		print(f'pass@k metrics:')
		for k, v in rollouts.meta_info['pass_at_k'].items():
			print(f'{k}: {v:.4f}')

# @hydra.main(version_base=None, config_path="../../configs", config_name="evaluate_api_llm")
# def main(config):
# 	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
# 	actor_wg = ApiCallingWrapperWg(config, tokenizer)
# 	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
# 	import time
# 	start_time = time.time()
# 	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}), val=True)
# 	print(f'[DEBUG] rollouts: {rollouts}')
# 	end_time = time.time()
# 	print(f'rollout time: {end_time - start_time} seconds')
# 	# print rollout rewards from the rm_scores
# 	rm_scores = rollouts.batch["rm_scores"]
# 	metrics = rollouts.meta_info["metrics"]
# 	avg_reward = rm_scores.sum(-1).mean().item()
# 	print(f'rollout rewards: {avg_reward}')
# 	print(f'metrics:')
# 	for k, v in metrics.items():
# 		print(f'{k}: {v}')



if __name__ == "__main__":
	main()
