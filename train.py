"""
Borrowed from verl.trainer.main_ppo.py
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from ragen.trainer.agent_trainer import RayAgentTrainer

import ray
import hydra
import os
from verl import DataProto
import torch
import numpy as np
from ragen.utils import register_resolvers
register_resolvers()
import sys

class DummyRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            score = data_item.non_tensor_batch['reward']
            score = float(score)
 
            reward_tensor[i, valid_response_length - 1] = score
            all_scores.append(score)

            # Get data_source from data_item if available, otherwise use a default value
            data_source = data_item.non_tensor_batch.get('data_source', 'default')
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        print(f"[DEBUG] all_scores: {all_scores}")
        print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor

def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    if spec is None:
        raise RuntimeError(f"Failed to create module spec from '{file_path}'")
        
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not function_name:
        raise ValueError("Function name not specified in custom_reward_function config")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)



def add_dependency(config):
    # 检查是否为评估模式（total_training_steps为0）
    if hasattr(config.trainer, 'total_training_steps') and config.trainer.total_training_steps == 0:
        # 在评估模式下，确保train_batch_size至少等于ppo_mini_batch_size
        # 使用val环境组数量替代train环境组数量来计算batch_size
        config.data.train_batch_size = max(
            config.es_manager.val.env_groups * config.es_manager.val.group_size,
            config.ppo_mini_batch_size if config.ppo_mini_batch_size is not None else 16
        )
    else:
        # 正常训练模式下的计算
        config.data.train_batch_size = config.es_manager.train.env_groups * config.es_manager.train.group_size
    
    if config.ppo_mini_batch_size is None:
        config.ppo_mini_batch_size = config.data.train_batch_size // 4
        print(f"config.ppo_mini_batch_size: {config.ppo_mini_batch_size}")

    config.actor_rollout_ref.actor.ppo_mini_batch_size = config.ppo_mini_batch_size
    config.critic.ppo_mini_batch_size = config.ppo_mini_batch_size

    if config.micro_batch_size_per_gpu is None:
        config.micro_batch_size_per_gpu = config.actor_rollout_ref.actor.ppo_mini_batch_size // config.trainer.n_gpus_per_node
        print(f"config.micro_batch_size_per_gpu: {config.micro_batch_size_per_gpu}")
        
    config.actor_rollout_ref.actor.micro_batch_size_per_gpu = config.micro_batch_size_per_gpu
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = config.micro_batch_size_per_gpu
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = config.micro_batch_size_per_gpu
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = config.micro_batch_size_per_gpu
    config.critic.ppo_micro_batch_size_per_gpu = config.micro_batch_size_per_gpu

    return config


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    config = add_dependency(config)
    print(f"config: {config}")

    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    # 检查是否为纯评估模式
    is_eval_only = hasattr(config.trainer, 'total_training_steps') and config.trainer.total_training_steps == 0
    if is_eval_only:
        print("运行纯评估模式，将优化内存使用...")
        
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
                "RAY_DEBUG": "legacy" # used here for simpler breakpoint()
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint

        # 检查是否为纯评估模式
        is_eval_only = hasattr(config.trainer, 'total_training_steps') and config.trainer.total_training_steps == 0
        
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        # elif config.actor_rollout_ref.actor.strategy == 'megatron':
        #     assert  config.actor_rollout_ref.actor.strategy == config.critic.strategy
        #     from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        #     from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        #     ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # 在评估模式下简化角色映射
        if is_eval_only:
            role_worker_mapping = {
                Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            }
            # 确保关闭任何参考策略
            config.actor_rollout_ref.actor.use_ref = False
            config.actor_rollout_ref.actor.use_kl_loss = False
        else:
            # 正常训练模式下的角色映射
            role_worker_mapping = {
                Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
                Role.Critic: ray.remote(CriticWorker),
            }
            if config.actor_rollout_ref.actor.use_ref:
                print("[DEBUG] using ref policy")
                role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            else:
                print("[DEBUG] not using ref policy, setting use_kl_loss to False")
                config.actor_rollout_ref.actor.use_kl_loss = False
        
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # 在评估模式下简化资源映射
        if is_eval_only:
            mapping = {
                Role.ActorRollout: global_pool_id,
            }
        else:
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
            }
            if config.actor_rollout_ref.actor.use_ref:
                mapping[Role.RefPolicy] = global_pool_id

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from ragen.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reward_manager_name = config.reward_model.get("reward_manager", "dummy")
        # print(f'reward_manager_name: {reward_manager_name}')
        # if reward_manager_name == 'dummy':
        print("using dummy reward manager")
        reward_manager_cls = DummyRewardManager
        # elif reward_manager_name == 'naive':
        #     from verl.workers.reward_manager import NaiveRewardManager
        #     reward_manager_cls = NaiveRewardManager
        # elif reward_manager_name == 'prime':
        #     from verl.workers.reward_manager import PrimeRewardManager
        #     reward_manager_cls = PrimeRewardManager
        # else:
        #     raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayAgentTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
        
        # 在评估模式下优化初始化流程
        if is_eval_only:
            print("评估模式：初始化ActorRollout worker并执行评估...")
            trainer.init_eval_only_workers()
            trainer.evaluate()
        else:
            # 正常训练流程
            trainer.init_workers()
            trainer.init_agent_proxy()
            trainer.fit()


if __name__ == '__main__':
    main()
