import copy
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
 
import random

from ..core.object import Agent
from ..actions import *
from ..core.room import Room
from .spatial_solver import SpatialSolver

@dataclass
class ExplorationTurnLog:
    """Log data for a single exploration turn."""
    node_coverage: float
    edge_coverage: float
    step: int
    action_counts: Dict[str, int]
    observed_items: List[str]
    visible_objects: List[str]
    is_action_fail: bool = False
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None
    information_gain: Optional[float] = None  # Information gain (uses exploration quality metric)
    possible_positions: Optional[Dict[str, List[List[int]]]] = None  # Sampled possible positions per object

    def to_dict(self):
        return {
            "node_coverage": self.node_coverage,
            "edge_coverage": self.edge_coverage,
            "step": self.step,
            "observed_items": self.observed_items,
            "visible_objects": self.visible_objects,
            "is_action_fail": self.is_action_fail,
            "action_counts": dict(self.action_counts),
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "information_gain": self.information_gain or 0.0,
            "possible_positions": self.possible_positions or {},
        }

class ExplorationManager:
    """Minimal exploration manager without graphs.

    - Keeps copies of `room` and `agent` for simulation.
    - Executes actions and logs turns.
    - Graph-related metrics default to safe zeros.
    """
    MAX_POSSIBLE_POSITIONS_PER_OBJECT: int = 200
    DEFAULT_ACTION_COUNTS = {'move': 0, 'rotate': 0, 'return': 0, 'observe': 0, 'term': 0, 'forced_term': 0, 'query': 0}
    def __init__(self, room: Room, agent: Agent, grid_size: int | None = None):
        self.base_room = room.copy()
        self.exploration_room = room.copy()
        self.agent = agent.copy()
        self.keep_object_names = [self.agent.name] + [obj.name for obj in getattr(self.exploration_room, 'all_objects', [])]

        self.turn_logs: List[ExplorationTurnLog] = []
        # History now stores ActionResult for each executed action (in order)
        self.history: List['ActionResult'] = []
        
        # Coverage tracking (exclude gates)
        self._init_node_name = "initial_pos"
        self.init_pos = self.agent.init_pos.copy()
        self._init_room_id = int(self.agent.init_room_id)

        # Node names: all objects in the exploration room
        self.node_names: List[str] = [o.name for o in self.exploration_room.all_objects]

        # Edge targets: per-room object pairs + (init, object-in-init-room)
        self.target_edges: Set[frozenset] = set()
        for rid, names in self.exploration_room.objects_by_room.items():
            names += self.exploration_room.gates_by_room.get(rid, [])
            if not names:
                continue
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    self.target_edges.add(frozenset({a, b}))
        for name in self.exploration_room.objects_by_room[self._init_room_id] + self.exploration_room.gates_by_room.get(self._init_room_id, []):
            self.target_edges.add(frozenset({self._init_node_name, name}))
        
        self.observed_nodes: Set[str] = set()
        self.known_edges: Set[frozenset] = set()

        # Action counts and costs
        self.action_counts: Dict[str, int] = self.DEFAULT_ACTION_COUNTS.copy()
        self.action_cost: int = 0
        # Observed names (objects and gates) to gate JumpTo() eligibility
        self.observed_items: Set[str] = set()
        self.visible_objects: List[str] = []
        # Grid size for solver metrics (use provided or infer from mask; fallback 10)
        inferred_g = (max(self.exploration_room.mask.shape) if getattr(self.exploration_room, 'mask', None) is not None else 10)
        self.grid_size: int = int(inferred_g if grid_size is None else grid_size)
        # Spatial solver for info gain / quality
        self.spatial_solver = SpatialSolver(self.node_names + ['initial_pos'], self.grid_size)
        self.spatial_solver.set_initial_position('initial_pos', (0, 0))
        
    def _execute_and_update(self, action: BaseAction, **kwargs) -> ActionResult:
        """Execute action and update exploration state."""
        # Enforce "observed-before-move"
        if isinstance(action, MoveAction):
            kwargs['observed_items'] = list(self.observed_items)
        result = action.execute(self.exploration_room, self.agent, **kwargs)
        # Log every action result to history immediately
        self.history.append(result)
        if not result.success:
            return result
        
        # Count action, cost, and update coverage
        self.action_counts[result.action_type] = self.action_counts.get(result.action_type, 0) + 1
        self.action_cost += int(action.cost)
        if isinstance(action, ObserveAction):
            self._update_coverage_from_observe(result)
        
        return result



    def execute_action(self, action: BaseAction) -> ActionResult:
        """Execute single action and return result."""
        return self._execute_and_update(action)
    
    def execute_success_action(self, action: BaseAction, **kwargs) -> ActionResult:
        """Execute single action and return result (must be successful)."""
        result = self._execute_and_update(action, **kwargs)
        assert result.success, f"Action {action} with kwargs {kwargs} failed: {result.message}"
        return result

    def execute_action_sequence(self, action_sequence: ActionSequence) -> List[ActionResult]:
        """
        Execute a sequence of motion actions followed by a final action.
        If any motion action fails, execute an observe action and end.
        Returns list of action results.
        """
        assert action_sequence.final_action, "Action sequence requires a final action."

        action_results = []
        is_action_fail = False
        # Execute motion actions
        for action in action_sequence.motion_actions:
            result = self._execute_and_update(action)
            action_results.append(result)
            if not result.success:
                is_action_fail = True
                # On failure, perform an observe action and end
                obs_result = self._execute_and_update(ObserveAction())
                obs_result.message = f"Subsequent actions are skipped due to failure, instead an observe is executed: {obs_result.message}"
                action_results.append(obs_result)
                assert obs_result.success, f"Observe action failed: {obs_result.message}"
                self._log_exploration(action_results, is_action_fail)
                return action_results

        # Execute final action
        final_action = action_sequence.final_action
        result = self._execute_and_update(final_action)
        action_results.append(result)
        if not result.success:
            is_action_fail = True
        # Always log before return
        self._log_exploration(action_results, is_action_fail)
        return action_results
    
    def finish_exploration(self, return_to_origin: bool = True) -> Room:
        """Complete exploration and return final room state."""
        if return_to_origin:
            result = self.execute_action(ReturnAction())
            if not result.success:
                raise ValueError(f"Failed to return to origin: {result.message}")
        return self.exploration_room
    
    def get_exp_summary(self) -> Dict[str, Any]:
        """Get exploration summary."""
        node_cov = len(self.observed_nodes) / len(self.node_names)
        edge_cov = len(self.known_edges) / len(self.target_edges)
        info_gain_list = [turn_log.information_gain for turn_log in self.turn_logs] if self.turn_logs else []
        acc_info_gain = sum(info_gain_list)
        avg_info_gain = acc_info_gain / len(self.turn_logs) if self.turn_logs else 0.0
        return {
            "node_coverage": node_cov,
            "edge_coverage": edge_cov,
            "n_exploration_steps": len(self.turn_logs),
            "action_counts": dict(self.action_counts),
            "action_cost": int(self.action_cost),
            "exploration_cost": int(self.action_cost),
            "info_gain_list": info_gain_list,
            "acc_info_gain": acc_info_gain,
            "avg_info_gain": avg_info_gain,
        }
    
    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict] = None) -> Dict[str, Any]:
        """Calculate exploration performance for a group from env_data_list.
        Prefer using precomputed per-sample metrics if present.
        """
        if not env_data_list:
            return {}

        pre = [((s.get('metrics') or {}).get('exploration') or {}) for s in env_data_list]

        def _avg_key(k: str) -> float:
            vals = [p.get(k) for p in pre if isinstance(p.get(k), (int, float))]
            return (sum(vals) / len(vals)) if vals else 0.0

        result = {
            'avg_node_coverage': _avg_key('last_node_coverage'),
            'avg_edge_coverage': _avg_key('last_edge_coverage'),
            'avg_exploration_steps': _avg_key('n_exploration_steps'),
            'avg_action_cost': _avg_key('action_cost'),
            'avg_action_fail_ratio': _avg_key('action_fail_ratio'),
            'avg_valid_action_ratio': _avg_key('valid_action_ratio'),
            'avg_final_information_gain': _avg_key('final_information_gain'),
            'infogain_per_turn': ExplorationManager._avg_lists_carry_forward([p.get('information_gain_per_turn') or [] for p in pre]),
        }

        # Average action counts
        agg_counts: Dict[str, float] = {}
        n = len(pre)
        for p in pre:
            for a, c in (p.get('action_counts') or {}).items():
                agg_counts[a] = agg_counts.get(a, 0.0) + float(c)
        if agg_counts:
            for a in list(agg_counts.keys()):
                agg_counts[a] /= n
            result['avg_action_counts'] = agg_counts
        return result
    
    @staticmethod
    def _calculate_infogain_per_turn(env_data_list: List[Dict]) -> List[float]:
        """Calculate average information gain for each turn across all samples."""
        # Collect all turn information gains by turn index
        turn_infogains = defaultdict(list)  # turn_index -> list of infogain values
        PAD = None
        
        for env_data in env_data_list:
            env_turn_logs = env_data.get('env_turn_logs', [])
            for turn_idx, turn_log in enumerate(env_turn_logs):
                # Only consider exploration phases
                if turn_log.get('is_exploration_phase', False):
                    infogain = turn_log.get('exploration_log', {}).get('information_gain')
                    if infogain is not None:
                        turn_infogains[turn_idx].append(infogain)
        
        # Calculate averages for each turn
        max_turns = max(turn_infogains.keys()) if turn_infogains else -1
        avg_infogains = []
        
        for turn_idx in range(max_turns + 1):
            if turn_idx in turn_infogains and turn_infogains[turn_idx]:
                avg_infogain = sum(turn_infogains[turn_idx]) / len(turn_infogains[turn_idx])
                avg_infogains.append(avg_infogain)
            else:
                # carry forward last average if available, else 0.0
                if avg_infogains:
                    avg_infogains.append(avg_infogains[-1])
                else:
                    avg_infogains.append(0.0)
        
        return avg_infogains

    @staticmethod
    def _avg_lists_carry_forward(list_of_lists: List[List[float]]) -> List[float]:
        if not list_of_lists:
            return []
        max_len = max((len(lst) for lst in list_of_lists), default=0)
        if max_len == 0:
            return []
        padded: List[List[float]] = []
        for lst in list_of_lists:
            if not lst:
                padded.append([0.0] * max_len)
                continue
            last = lst[-1]
            if len(lst) < max_len:
                lst = lst + [last] * (max_len - len(lst))
            padded.append(lst)
        out: List[float] = []
        for i in range(max_len):
            vals = [lst[i] for lst in padded if isinstance(lst[i], (int, float))]
            out.append((sum(vals) / len(vals)) if vals else 0.0)
        return out

    @staticmethod
    def aggregate_per_sample(env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate exploration metrics within a single sample.
        - node/edge coverage (last)
        - total action cost
        - action counts
        - per-turn information gain and final information gain
        - exploration steps
        - is_action_fail and is_valid_action proportions
        """
        env_turn_logs = env_data.get('env_turn_logs', [])
        last_exp = None
        for t in reversed(env_turn_logs):
            if t.get('is_exploration_phase', False) and t.get('exploration_log'):
                last_exp = t['exploration_log']
                break
        node_cov = last_exp.get('node_coverage', 0.0) if last_exp else 0.0
        edge_cov = last_exp.get('edge_coverage', 0.0) if last_exp else 0.0
        steps = last_exp.get('step', 0) if last_exp else 0
        # Approximate action counts and cost: derive from last turn summary fields if present
        default_counts = ExplorationManager.DEFAULT_ACTION_COUNTS.copy()
        action_counts = (last_exp.get('action_counts') if last_exp and ('action_counts' in last_exp) else {}) or {}
        # ensure default keys exist
        for k in default_counts:
            action_counts[k] = int(action_counts.get(k, 0))
        # Compute action cost if not present using known costs
        action_cost = last_exp.get('action_cost') if last_exp and ('action_cost' in last_exp) else None
        if action_cost is None and action_counts:
            # default costs aligned with action classes
            default_costs = {
                'move': 0,
                'rotate': 0,
                'return': 0,
                'observe': 1,
                'term': 0,
                'forced_term': 0,
                'query': 2,
            }
            action_cost = 0
            for k, v in action_counts.items():
                c = default_costs.get(k.lower(), 0)
                try:
                    action_cost += int(v) * int(c)
                except Exception:
                    continue
        if action_cost is None:
            action_cost = 0
        info_gain_list = last_exp.get('info_gain_list') if last_exp else None
        if info_gain_list is None:
            # rebuild from per-turn logs
            info_gain_list = []
            for t in env_turn_logs:
                if t.get('is_exploration_phase', False):
                    ig = (t.get('exploration_log') or {}).get('information_gain')
                    if ig is not None:
                        info_gain_list.append(ig)
        final_infogain = (info_gain_list[-1] if info_gain_list else 0.0)

        # Calculate proportions of is_action_fail and is_valid_action across all turns
        total_turns = len(env_turn_logs)
        action_fail_count = 0
        valid_action_count = 0

        for t in env_turn_logs:
            # Count is_action_fail from exploration_log
            if t.get('is_exploration_phase', False) and t.get('exploration_log'):
                if t['exploration_log'].get('is_action_fail', False):
                    action_fail_count += 1

            # Count is_valid_action from info
            if t.get('info', {}).get('is_valid_action', True):  # Default to True if not present
                valid_action_count += 1

        action_fail_ratio = action_fail_count / total_turns if total_turns > 0 else 0.0
        valid_action_ratio = valid_action_count / total_turns if total_turns > 0 else 0.0

        return {
            'last_node_coverage': node_cov,
            'last_edge_coverage': edge_cov,
            'n_exploration_steps': steps,
            'action_counts': action_counts,
            'action_cost': action_cost,
            'information_gain_per_turn': info_gain_list or [],
            'final_information_gain': final_infogain,
            'action_fail_ratio': action_fail_ratio,
            'valid_action_ratio': valid_action_ratio,
        }
    
    # No passive history generation here; proxies produce text histories directly.
    
    # === Coverage helpers ===
    def _anchor_name(self) -> Optional[str]:
        # If standing on an object position, use that object as anchor (exclude gates)
        for obj in self.exploration_room.all_objects:
            if np.allclose(obj.pos, self.agent.pos):
                return obj.name
        # Initial position anchor
        if np.allclose(self.agent.pos, self.init_pos):
            return self._init_node_name
        raise ValueError("No anchor found")

    def _update_coverage_from_observe(self, observe_result: 'ActionResult') -> None:
        visible = observe_result.data.get('visible_objects', []) or []
        # node coverage
        for name in visible:
            self.observed_items.add(name)
            if name in self.node_names:
                self.observed_nodes.add(name)
        # edge coverage: observe A from B (B is anchor)
        anchor = self._anchor_name()
        for name in visible:
            if name == anchor:
                continue
            pair = frozenset({anchor, name})
            if pair in self.target_edges:
                self.known_edges.add(pair)

    def _update_coverage_from_query(self, query_result: 'ActionResult') -> None:
        # Coverage: two nodes + edge between them
        objs = query_result.data.get('objects') or query_result.data.get('pair') or []
        if len(objs) == 2:
            a, b = objs[0], objs[1]
            if a in self.node_names:
                self.observed_nodes.add(a)
            if b in self.node_names:
                self.observed_nodes.add(b)
            pair = frozenset({a, b})
            if pair in self.target_edges:
                self.known_edges.add(pair)


    
    def _log_exploration(self, action_results: List['ActionResult'], is_action_fail = False) -> None:
        """Log exploration history and efficiency."""
        # First ingest latest observations, then compute info gain as exploration quality
        for ar in action_results:
            if getattr(ar, 'action_type', None) in ('observe'):
                self.visible_objects = ar.data.get('visible_objects', []) or []
            self._calculate_single_action_information_gain(ar)
        turn_quality = self._compute_exploration_quality()
        # Snapshot possible positions per object (only initialized/observed ones), with sampling
        possible_positions = self._get_possible_positions_snapshot(self.MAX_POSSIBLE_POSITIONS_PER_OBJECT)
        
        step_idx = len(self.turn_logs) + 1
        turn_log = ExplorationTurnLog(
            node_coverage=len(self.observed_nodes) / len(self.node_names),
            edge_coverage=len(self.known_edges) / len(self.target_edges),
            observed_items=list(self.observed_items),
            visible_objects=self.visible_objects,
            step=step_idx,
            is_action_fail=is_action_fail,
            action_counts=self.action_counts,
            room_state=self.exploration_room.copy(),
            agent_state=self.agent.copy(),
            information_gain=turn_quality if turn_quality is not None else (self.turn_logs[-1].information_gain if self.turn_logs else 0.0),
            possible_positions=possible_positions,
        )
        self.turn_logs.append(turn_log)
    

    def _get_possible_positions_snapshot(self, max_per_obj: Optional[int] = None) -> Dict[str, List[List[int]]]:
        """Return sampled possible positions for each observed object (exclude 'initial_pos')."""
        try:
            variables = self.spatial_solver.solver.variables if self.spatial_solver else {}
        except Exception:
            return {}
        snapshot: Dict[str, List[List[int]]] = {}
        for name, var in variables.items():
            if name == 'initial_pos':
                continue
            # Only include objects that have been observed at least once
            if name not in self.observed_items:
                continue
            dom = getattr(var, 'domain', None)
            if dom is None or len(dom) == 0:
                continue
            pts = list(dom)
            if max_per_obj is not None and len(pts) > int(max_per_obj):
                pts = random.sample(pts, int(max_per_obj))
            snapshot[name] = [list(p) for p in pts]
        return snapshot
    
    def _calculate_single_action_information_gain(self, action_result: 'ActionResult') -> float:
        """Ingest observation/query triples into the solver; return 0.0.
        Keep simple: we use exploration quality as info gain elsewhere.
        """
        if getattr(action_result, 'action_type', None) in ('observe', 'query'):
            triples = action_result.data.get('relation_triples', []) if hasattr(action_result, 'data') else []
            if triples:
                keep = set(self.spatial_solver.solver.variables.keys())
                filt = [tr for tr in triples if tr.subject in keep and tr.anchor in keep]
                if filt:
                    self.spatial_solver.add_observation(filt)

    # === Exploration quality helpers ===
    def _full_grid_cell_count(self) -> int:
        return int(self.grid_size) * int(self.grid_size)

    def _final_position_counts(self) -> Dict[str, int]:
        """Counts of possible positions per variable at the end of exploration.
        Uses existing solver if available, otherwise rebuilds a solver from history.
        """
        if self.spatial_solver is not None:
            return self.spatial_solver.get_num_possible_positions()
        # Build a temporary solver and ingest history triples
        solver = SpatialSolver(self.node_names + ['initial_pos'], self.grid_size)
        solver.set_initial_position('initial_pos', (0, 0))
        for ar in self.history:
            try:
                if getattr(ar, 'action_type', None) in ('observe', 'query'):
                    triples = ar.data.get('relation_triples', []) if hasattr(ar, 'data') else []
                    if triples:
                        solver.add_observation(triples)
            except Exception:
                continue
        return solver.get_num_possible_positions()

    def _compute_exploration_quality(self) -> float | None:
        """Compute quality = sum_i log2(M/Ci) / (N * log2(M)). Exclude 'initial_pos'. Include gates.
        Returns None if computation is not applicable.
        """
        try:
            counts = self._final_position_counts()
            M = self._full_grid_cell_count()
            if M <= 1:
                return 0.0
            names = [n for n in counts.keys() if n != 'initial_pos']
            if not names:
                return 0.0
            denom = len(names) * np.log2(M)
            if denom <= 0:
                return 0.0
            total = 0.0
            for n in names:
                Ci = max(1, int(counts.get(n, M)))
                total += float(np.log2(M / Ci))
            return float(total / denom)
        except Exception:
            return None

if __name__ == "__main__":
    pass