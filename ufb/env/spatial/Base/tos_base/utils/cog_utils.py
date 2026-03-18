"""
Cognitive Map Utility Functions

This module provides utility functions for evaluating cognitive maps using turn logs
and LLM interfaces.
"""

import numpy as np
from typing import List, Dict, Any
from verl import DataProto
from ..managers.cognitive_map_manager import CognitiveMapManager
from ..managers.history_manager import HistoryManager
from ..prompts.cogmap_prompts import get_cogmap_prompt
from .. import Room, Agent
import re
from tqdm import tqdm

def _evaluate_cogmaps(
    cognitive_map_manager: CognitiveMapManager,
    responses_by_type: Dict[str, str],
    turn_log: Dict[str, Any],
):
    room_state = Room.from_dict(turn_log['room_state'])
    agent_state = Agent.from_dict(turn_log['agent_state'])

    # Determine observed items
    if turn_log.get('is_exploration_phase'):
        observed_items = turn_log.get('exploration_log',{}).get('observed_items', [obj.name for obj in room_state.all_objects])
    elif (not turn_log.get('is_exploration_phase', True)) and 'falsebelief' in turn_log.get('evaluation_log', {}).get('task_type', '').lower():
        observed_items = [turn_log.get('evaluation_log', {}).get('evaluation_data', {}).get('kwargs', {}).get('rotated_object')]
    else:
        raise ValueError("turn_log must be either exploration phase or false belief evaluation")

    # Evaluate cognitive maps using selected responses
    return cognitive_map_manager.evaluate_cogmaps(
        responses_by_type,
        room_state,
        agent_state,
        observed_items
    )

def evaluate_cognitive_maps_from_turnlogs(
    env_summarys: List[Dict[str, Any]],  # List of env summaries from get_env_summary()
    message_lists: List[List[Dict[str, str]]],  # Corresponding list of message histories
    llm_wrapper,
    override_cogmap: bool = False,
    cogmap_config: Dict[str, Any] = None,
    vagen: bool = False,  # Whether to use _generate_batch_responses interface
    reevaluate: bool = False,  # Whether to re-evaluate existing cogmaps
) -> None:
    """
    Evaluate cognitive maps for each turn in multiple environments' turn logs.

    This function processes all environments' turn logs, constructs message histories
    with COGMAP_EXP_REQUIRED_INSTRUCTION, makes batch LLM calls for all inputs,
    evaluates cognitive maps using CognitiveMapManager, and stores results back
    into the corresponding env turn logs.

    Args:
        env_summarys: List of environment summary dicts from get_env_summary(),
                     each containing 'env_turn_logs' and 'env_info' keys
        message_lists: List of message histories for each environment
        llm_wrapper: ApiCallingWrapperWg instance for LLM API calls
        override_cogmap: If True, override existing cogmap responses
        cogmap_config: Configuration for cognitive map manager
        vagen: If True, use _generate_batch_responses interface;
                          if False, use _call_llm_batch interface
    """
    assert len(env_summarys) == len(message_lists), "env_summarys and message_lists must have the same length"
    if not cogmap_config:
        cogmap_config = {
            "cogmap_type": "standard",
            "pos_allow_scale": False,
            "scope": "all",
        }

    # Collect all message sequences and metadata
    all_messages_list = []
    all_env_ids = []
    env_id_to_location = {}  # Mapping from env_id to (env_idx, turn_idx, map_type)
    env_cogmap_managers: Dict[int, CognitiveMapManager] = {}  # Store CognitiveMapManager for each env
    env_history_managers: Dict[int, HistoryManager] = {}  # Store HistoryManager for each env

    env_id_counter = 0
    for env_idx, env_summary in enumerate(env_summarys):
        env_info = env_summary['env_info']
        env_config = env_info['config']
        room_config = env_info['initial_room']
        agent_config = env_info['initial_agent']
        # Create CognitiveMapManager
        cognitive_map_manager = CognitiveMapManager(
            **cogmap_config
        )
        env_cogmap_managers[env_idx] = cognitive_map_manager

        history_manager = HistoryManager(env_config['observation_config'], env_config['model_config'],room_config, agent_config, env_config['output_dir'])
        env_history_managers[env_idx] = history_manager

        turn_logs = env_summary['env_turn_logs']
        env_messages = [msg for msg in message_lists[env_idx] if msg.get('role') != 'system']

        if env_config.get('exp_type') == 'active':
            for turn_idx, turn_log in enumerate(turn_logs):
                if turn_idx==0 or not turn_logs[turn_idx-1].get('info', {}).get('is_valid_action', False):
                    continue
                messages = []
                # Include all messages up to and including current turn
                max_message_idx = min((turn_idx+1) * 2, len(env_messages))
                for msg_idx in range(max_message_idx):
                    msg = env_messages[msg_idx]
                    messages.append(msg.copy())

                # Check if cogmap response already exists in history
                if turn_log['is_exploration_phase']:
                    target_turn_idx = turn_idx - 1
                    cogmap = history_manager.get_cogmap(target_turn_idx)
                    if cogmap and not override_cogmap:
                        if reevaluate:
                            responses_by_type = {k: v['original_response'] for k, v in cogmap.items() if v.get('original_response')}
                            cogmap_log = _evaluate_cogmaps(cognitive_map_manager, responses_by_type, turn_logs[target_turn_idx])
                            # Write reevaluated cogmap back to the correct (target) turn log
                            turn_logs[target_turn_idx]['cogmap_log'] = cogmap_log.to_dict() if cogmap_log else {}
                            history_manager.update_cogmap(turn_logs[target_turn_idx])
                        continue
                    assert messages[-2]["role"] == "user", f"Expected user message but got {messages[-2]['role']}"

                    # Active exploration:
                    # - For each turn: generate local and global
                    # - For final turn only: also generate rooms and relations
                    base_user = re.sub(r"You have a maximum of\s*\d+\s*exploration steps left.*", "", turn_log['user_message'], flags=re.DOTALL)
                    if len(turn_logs[target_turn_idx]['exploration_log']['visible_objects']) == 0:
                        per_turn_types = ['global']
                    else:
                        per_turn_types = ['local', 'global']
                    final_only_types = ['rooms'] if turn_log['is_last_exp'] else []
                    for map_type in per_turn_types + final_only_types:
                        msgs = [m.copy() for m in messages]
                        enable_think = bool(env_config.get('prompt_config', {}).get('enable_think', True))
                        cogmap_prompt = get_cogmap_prompt(map_type, enable_think)
                        msgs[-2]["content"] = base_user + cogmap_prompt
                        msgs.pop()
                        env_id_to_location[env_id_counter] = (env_idx, target_turn_idx, map_type)
                        all_messages_list.append(msgs)
                        all_env_ids.append(env_id_counter)
                        env_id_counter += 1

                # only effective for false belief task
                elif not turn_log['is_exploration_phase'] and 'falsebelief' in turn_log.get('evaluation_log', {}).get('task_type', '').lower():
                    assert turn_log.get('evaluation_log', {}).get('evaluation_data', {}).get('action')
                    assert messages[-2]["role"] == "user", f"Expected user message but got {messages[-2]['role']}"
                    target_turn_idx = turn_idx
                    base_user = re.sub(r'## Evaluation Question.*', '', turn_log['user_message'],  flags=re.DOTALL) + turn_log['evaluation_log']['evaluation_data']['action']

                    map_type = 'false_belief'
                    msgs = [m.copy() for m in messages]
                    enable_think = bool(env_config.get('prompt_config', {}).get('enable_think', True))
                    cogmap_prompt = get_cogmap_prompt(map_type, enable_think)
                    msgs[-2]["content"] = base_user + cogmap_prompt
                    msgs.pop()
                    env_id_to_location[env_id_counter] = (env_idx, target_turn_idx, map_type)
                    all_messages_list.append(msgs)
                    all_env_ids.append(env_id_counter)
                    env_id_counter += 1
                else:
                    continue

        elif env_config.get('exp_type') == 'passive':
            # Check if cogmap response already exists in history
            if history_manager.get_cogmap(0) and not override_cogmap:
                continue

            assert env_messages[0]["role"] == "user", f"Expected user message but got {env_messages[0]['role']}"
            base_user = re.sub(r'## Evaluation Question.*', '', turn_logs[0].get('user_message', ''), flags=re.DOTALL)

            # Passive: only global for first exp turn (correctness only)
            map_type = 'global'
            messages = [env_messages[0].copy()]
            enable_think = bool(env_config.get('prompt_config', {}).get('enable_think', True))
            cogmap_prompt = get_cogmap_prompt(map_type, enable_think)
            messages[0]["content"] = base_user + cogmap_prompt
            all_messages_list.append(messages)
            all_env_ids.append(env_id_counter)
            env_id_to_location[env_id_counter] = (env_idx, 0, map_type)
            env_id_counter += 1
            # let cogmaplog be reused in passive
            turn_log = {'turn_number': 1 ,"user_message": base_user, "is_exploration_phase": True, "room_state": room_config, "agent_state": agent_config}
            turn_logs[0] = turn_log
            if not history_manager.has_exploration(0):
                history_manager.update_turn_log(turn_log)


    if all_messages_list:
        response_texts = _call_llm_batch(
            llm_wrapper, all_messages_list, all_env_ids, vagen
        )

        # Group responses by (env_idx, turn_idx)
        responses_by_turn = {}
        for response, original_env_id in zip(response_texts, all_env_ids):
            env_idx, turn_idx, map_type = env_id_to_location[original_env_id]
            key = (env_idx, turn_idx)
            if key not in responses_by_turn:
                responses_by_turn[key] = {}
            responses_by_turn[key][map_type] = response

        # Process each turn with its grouped responses
        for (env_idx, turn_idx), responses_by_type in responses_by_turn.items():
            env_summary = env_summarys[env_idx]
            turn_log = env_summary['env_turn_logs'][turn_idx]
            cognitive_map_manager = env_cogmap_managers[env_idx]
            history_manager = env_history_managers[env_idx]
            env_config = env_summary['env_info']['config']

            assert responses_by_type and turn_log.get('room_state') and turn_log.get('agent_state')

            # Evaluate cognitive maps using selected responses
            try:
                cogmap_log = _evaluate_cogmaps(cognitive_map_manager,responses_by_type,turn_log)
                turn_log['cogmap_log'] = cogmap_log.to_dict() if cogmap_log else {}
                history_manager.update_cogmap(turn_log)
            except Exception as e:
                print(f"Error evaluating cognitive map: {e}")

    for env_idx, history_manager in env_history_managers.items():
        history_manager.save()

    return env_summarys

def _call_llm_batch(
    llm_wrapper,
    all_messages_list: List[List[Dict[str, str]]],
    all_env_ids: List[int],
    vagen: bool = False
) -> List[str]:
    """
    Helper function to make batch LLM calls using ApiCallingWrapperWg.

    Args:
        llm_wrapper: The LLM wrapper instance
        all_messages_list: List of message lists for each request
        all_env_ids: List of environment IDs (used for ordering)
        vagen: If True, use generate interface

    Returns:
        List of response texts in the same order as input
    """
    CHUNK_SIZE = 128
    total = len(all_messages_list)

    def run_chunk(msgs_chunk, ids_chunk):
        if vagen:
            return [r['text'] for r in llm_wrapper.generate(msgs_chunk)]
        lm_inputs = DataProto()
        lm_inputs.non_tensor_batch = {
            'messages_list': np.array(msgs_chunk, dtype=object),
            'env_ids': np.array(ids_chunk, dtype=object),
            'group_ids': np.array(ids_chunk, dtype=object)
        }
        lm_outputs = llm_wrapper.generate_sequences(lm_inputs)
        return lm_outputs.non_tensor_batch['response_texts']

    results_by_id = {}
    with tqdm(total=total, desc='COGMAP') as pbar:
        for start in range(0, total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total)
            msgs_chunk = all_messages_list[start:end]
            ids_chunk = all_env_ids[start:end]
            texts = run_chunk(msgs_chunk, ids_chunk)
            for eid, text in zip(ids_chunk, texts):
                results_by_id[eid] = text
            pbar.update(len(texts))

    return [results_by_id[eid] for eid in all_env_ids]