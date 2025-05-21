
from ragen.llm_agent.ctx_manager import ContextManager
from ragen.llm_agent.es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from ragen.llm_agent.base_llm import ConcurrentLLM
from ragen.llm_agent.agent_proxy import ApiCallingWrapperWg, VllmWrapperWg, LLMAgentProxy

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config):
	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
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

if __name__ == "__main__":
	main()
