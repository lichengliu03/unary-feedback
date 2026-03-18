"""
Cognitive Map Manager

Minimal, modular evaluator for cognitive maps.

Responsibilities:
- Extract JSON from LLM response
- Transform JSON sections (global/local/rooms/gates) into BaseRoom-compatible data
- Evaluate global, local, room maps (dir/facing/pos) using consistent coordinates
- Evaluate gates connectivity
- Log all results per turn for summary aggregation
"""

import json
import re
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import copy
from ..actions.base import BaseAction
from ..core.room import Room, BaseRoom
from ..core.object import Object, Agent, Gate
from ..utils.relationship_utils import room_to_ordered_relations        
from ..utils.relation_codes import decode_relation_codes,invert_pair_key, invert_dir_code
# Utils
from ..utils.cogmap.transforms import (
    transform_baseroom,
    br_from_anchor_to_initial,
)
from ..utils.cogmap.metrics import compute_map_metrics
from ..utils.cogmap.consistency import (
    local_vs_global_consistency,
    rooms_vs_global_consistency,
    map_vs_relations_consistency,
    relations_consistency,
    stability,
)
from ..utils.cogmap.types import BaseCogMetrics, MapCogMetrics, RelationMetrics, ConsistencySummary, AccuracyMetrics
from ..utils.cogmap.analysis import (
    get_last_exploration_cogmap,
    get_false_belief_metrics,
    avg_nested_dicts,
)


@dataclass
class BaseCogMapTurnLog:
    """Common fields for all cogmap types."""
    type: str
    extraction_success: bool = False
    original_response: str = ""
    pred_json: Dict[str, Any] = field(default_factory=dict)
    pred_room_state: Optional['BaseRoom'] = None
    metrics: BaseCogMetrics = field(default_factory=BaseCogMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "extraction_success": self.extraction_success,
            "original_response": self.original_response,
            "pred_json": self.pred_json,
            "pred_room_state": self.pred_room_state.to_dict() if self.pred_room_state else {},
            "metrics": (self.metrics.to_dict() if self.metrics.valid else {}),
        }

@dataclass
class GlobalCogMapTurnLog(BaseCogMapTurnLog):
    gt_room_state: Optional['BaseRoom'] = None
    gt_json: Dict[str, Any] = field(default_factory=dict)
    gt_room_state_full: Optional['BaseRoom'] = None
    gt_json_full: Dict[str, Any] = field(default_factory=dict)
    metrics_full: BaseCogMetrics = field(default_factory=BaseCogMetrics)
    metric_agent: BaseCogMetrics = field(default_factory=BaseCogMetrics)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "gt_room_state": self.gt_room_state.to_dict() if self.gt_room_state else {},
            "gt_json": self.gt_json,
            "gt_room_state_full": self.gt_room_state_full.to_dict() if self.gt_room_state_full else {},
            "gt_json_full": self.gt_json_full,
            "metrics_full": (self.metrics_full.to_dict() if self.metrics_full.valid else {}),
            "metric_agent": (self.metric_agent.to_dict() if self.metric_agent.valid else {}),
        })
        return out

@dataclass
class LocalCogMapTurnLog(BaseCogMapTurnLog):
    gt_room_state: Optional['BaseRoom'] = None
    gt_json: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "gt_room_state": self.gt_room_state.to_dict() if self.gt_room_state else {},
            "gt_json": self.gt_json,
        })
        return out

@dataclass
class RoomsCogMapTurnLog(BaseCogMapTurnLog):
    pred_rooms_state: Dict[str, 'BaseRoom'] = field(default_factory=dict)
    gt_rooms_state: Dict[str, 'BaseRoom'] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "pred_rooms_state": {k: (v.to_dict() if v is not None else {}) for k, v in (self.pred_rooms_state.items() if self.pred_rooms_state else [])},
            "gt_rooms_state": {k: (v.to_dict() if v is not None else {}) for k, v in (self.gt_rooms_state.items() if self.gt_rooms_state else [])},
        })
        return out


@dataclass
class RelationsCogMapTurnLog(BaseCogMapTurnLog):
    pred_relations: Dict[str, str] = field(default_factory=dict)
    gt_relations_full: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "pred_relations": self.pred_relations,
            "gt_relations_full": self.gt_relations_full,
        })
        return out


@dataclass
class CognitiveMapTurnLog:
    """Aggregate per-type logs for one turn."""
    global_log: Optional[GlobalCogMapTurnLog] = None
    local_log: Optional[LocalCogMapTurnLog] = None
    rooms_log: Optional[RoomsCogMapTurnLog] = None
    relations_log: Optional[RelationsCogMapTurnLog] = None
    false_belief_log: Optional[BaseCogMapTurnLog] = None
    consistency: Optional[ConsistencySummary] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.global_log:
            out["global"] = self.global_log.to_dict()
        if self.local_log:
            out["local"] = self.local_log.to_dict()
        if self.rooms_log:
            out["rooms"] = self.rooms_log.to_dict()
        if self.relations_log:
            out["relations"] = self.relations_log.to_dict()
        if self.false_belief_log:
            out["false_belief"] = self.false_belief_log.to_dict()
        if self.consistency:
            out["consistency"] = self.consistency.to_dict()
        return out


class CognitiveMapManager:
    """Evaluate cognitive map JSON against ground truth."""    
    def __init__(self, cogmap_type: str = "standard", pos_allow_scale: bool = False, scope: str = "all"):
        """Initialize cognitive map manager."""
        self.explore_logs: List[CognitiveMapTurnLog] = []
        self.evaluate_log: Optional[CognitiveMapTurnLog] = None

        self.config = {
            "cogmap_type": cogmap_type,
            "pos_allow_scale": bool(pos_allow_scale),
            "scope": (scope if scope in ("global", "all") else "all"),
        }
        # room_id -> first-entry gate name
        self.entry_gate_by_room: dict[int, str] = {}
        # position normalization scale (computed once in global frame)
        self._pos_norm_L: float | None = None
        self._start_room_id: int | None = None
        self._prev_room_id: int | None = None

    def get_supported_types(self) -> List[str]:
        return ["global", "local", "rooms", "relations"]

    def evaluate_cogmap_type(self, assistant_response: str, gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]], map_type: str) -> Optional[BaseCogMapTurnLog]:
        """Extract JSON and evaluate a single cogmap type (global|local|rooms|relations). Only compute what's needed for the given type."""
        self._register_active_entry_gate(gt_room)
        t = (map_type or "global").lower()
        json_dict = self._extract_json_from_text(assistant_response)
        if json_dict is None or gt_room is None:
            # Mark extraction failure as invalid metric of the appropriate type
            if t == "relations":
                m = RelationMetrics.invalid()
            elif t == "false_belief":
                m = AccuracyMetrics.invalid()
            else:
                m = MapCogMetrics.invalid()
            return BaseCogMapTurnLog(type=t, extraction_success=False, original_response=assistant_response, metrics=m)
        all_item_names = {o.name for o in gt_room.all_objects}
        observed_set: set[str] = set(all_item_names if observed_items is None else [str(x).replace('_', ' ') for x in observed_items])
        visible_names = self._visible_object_names(gt_room, gt_agent)

        if t == "global":
            pred_global_br = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent, map_type)
            gt_global_br = self._build_gt_global_baseroom(gt_room, gt_agent, observed_set)
            full_global = transform_baseroom(self._baseroom_from_gt(gt_room, gt_agent), gt_agent.init_pos, gt_agent.init_ori)
            agent_br = self._build_gt_global_agent_baseroom(gt_room, gt_agent)
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_global(pred_global_br, gt_global_br, full_global, agent_br, assistant_response, json_dict)
        
        if t == "false_belief":
            pred_global_br = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent, map_type)
            gt_global_br = self._build_gt_global_baseroom(gt_room, gt_agent, observed_set)
            return self._eval_false_belief(pred_global_br, gt_global_br, assistant_response, json_dict)

        if t == "local":
            pred_local_br = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent, map_type)
            gt_local_br = self._build_gt_local_baseroom(gt_room, gt_agent)
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_local(pred_local_br, gt_local_br, assistant_response, json_dict)

        if t == "rooms":
            pred_rooms_map = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent, map_type)
            gt_rooms_map = self._build_gt_room_baserooms(gt_room, gt_agent, observed_set)
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_rooms(pred_rooms_map, gt_rooms_map, assistant_response, json_dict)

        if t == "relations":
            # No map preprocessing needed; relations are a flat dict of pairs
            pred_relations = self._parse_predicted_relations(json_dict)
            full_global = transform_baseroom(self._baseroom_from_gt(gt_room, gt_agent), gt_agent.init_pos, gt_agent.init_ori)
            return self._eval_relations(pred_relations, full_global, assistant_response, json_dict)

        raise ValueError(f"Invalid map type: {t}")

    def _eval_global(self, pred_global_br: BaseRoom,  gt_global_br: BaseRoom, gt_room_state_full: BaseRoom, agent_br: BaseRoom, assistant_response: str, pred_json: Dict) -> GlobalCogMapTurnLog:
        gt_json = self.baseroom_to_json(gt_global_br, include_gates=True)
        metrics = self._compare_baserooms(pred_global_br, gt_global_br)
        gt_json_full = self.baseroom_to_json(gt_room_state_full, include_gates=True)
        metrics_full = self._compare_baserooms(pred_global_br, gt_room_state_full)
        metric_agent = self._compare_baserooms(pred_global_br, agent_br)
        return GlobalCogMapTurnLog(
            type="global",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_global_br,
            metrics=metrics,
            gt_room_state=gt_global_br,
            gt_json=gt_json,
            gt_room_state_full=gt_room_state_full,
            gt_json_full=gt_json_full,
            metrics_full=metrics_full,
            metric_agent=metric_agent,
        )

    def _eval_false_belief(self, pred_global_br: BaseRoom, gt_global_br: BaseRoom, assistant_response: str, pred_json: Dict) -> BaseCogMapTurnLog:
        """Evaluate false belief task - check if one object in observed_items has correct orientation."""
        # Create name-to-object mappings for comparison
        pred_objects = {o.name: o for o in pred_global_br.objects}
        gt_objects = {o.name: o for o in gt_global_br.objects}
        metrics = AccuracyMetrics(0.0)
        for name in gt_objects:
            gt_obj = gt_objects[name]
            pred_obj = pred_objects.get(name)

            # Only check objects that have orientation
            if gt_obj.has_orientation and pred_obj is not None:
                if np.array_equal(pred_obj.ori, gt_obj.ori):
                    metrics = AccuracyMetrics(1.0)
                    break
 
        return BaseCogMapTurnLog(
            type="false_belief",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_global_br,
            metrics=metrics,
        )

    def _eval_local(self, pred_local_br: BaseRoom, gt_local_br: BaseRoom, assistant_response: str, pred_json: Dict) -> LocalCogMapTurnLog:
        metrics = self._compare_baserooms(pred_local_br, gt_local_br)
        return LocalCogMapTurnLog(
            type="local",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_local_br,
            metrics=metrics,
            gt_room_state=gt_local_br,
            gt_json=self.baseroom_to_json(gt_local_br, include_gates=True),
        )

    def _eval_rooms(self, pred_rooms_map: Dict[str, BaseRoom], gt_rooms_map: Dict[int, BaseRoom], assistant_response: str, pred_json: Dict) -> RoomsCogMapTurnLog:
        per_room: List[MapCogMetrics] = []
        for rid in sorted(gt_rooms_map.keys()):
            gt_br = gt_rooms_map[rid]
            if len(gt_br.objects) == 0:
                continue
            pred_br = pred_rooms_map.get(str(rid)) or pred_rooms_map.get(rid) or BaseRoom(objects=[], name=f"pred_room_{rid}")
            per_room.append(self._compare_baserooms(pred_br, gt_br))
        metrics = MapCogMetrics.average(per_room)
        pred_rooms_state = {str(rid): br for rid, br in pred_rooms_map.items()}
        gt_rooms_state = {str(rid): br for rid, br in gt_rooms_map.items()}
        return RoomsCogMapTurnLog(
            type="rooms",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=None,
            metrics=metrics,
            pred_rooms_state=pred_rooms_state,
            gt_rooms_state=gt_rooms_state,
        )
    
    @staticmethod
    def _relations_accuracies(pred: Dict[str, str], gt: Dict[str, str]) -> Tuple[float, float, float]:
        if not pred or not gt:
            return 0.0, 0.0, 0.0
        
        dir_correct = dist_correct = 0
        
        # Get all unique pairs from both predicted and ground truth
        from ..utils.relation_codes import parse_pair_key, make_ordered_pair_key
        all_pairs = set()
        for key in list(gt.keys()) + list(pred.keys()):
            a, b = parse_pair_key(key)
            if a and b:
                canonical_key = make_ordered_pair_key(*sorted([a, b]))
                all_pairs.add(canonical_key)
        
        tot = len(all_pairs)
        
        for canonical_pair in all_pairs:
            a, b = parse_pair_key(canonical_pair)
            key_ab, key_ba = f"{a}|{b}", f"{b}|{a}"
            
            # Find GT relation (should be in canonical form)
            gt_rel = gt.get(canonical_pair)
            if gt_rel is None:
                continue  # Skip if no GT for this pair
                
            # Find predicted relation (could be either direction)
            pred_rel = pred.get(key_ab)
            needs_inversion = False
            if pred_rel is None:
                pred_rel = pred.get(key_ba)
                needs_inversion = True
            if pred_rel is None:
                continue  # Skip if no prediction for this pair
                
            # If predicted relation is in opposite order, invert it
            if needs_inversion:
                from ..utils.relation_codes import invert_relation_codes_str
                pred_rel = invert_relation_codes_str(pred_rel)
            
            gt_dir, gt_dist = decode_relation_codes(gt_rel)
            pred_dir, pred_dist = decode_relation_codes(pred_rel)
            
            if pred_dir == gt_dir:
                dir_correct += 1
            if pred_dist == gt_dist:
                dist_correct += 1
        
        if tot == 0:
            return 0.0, 0.0, 0.0
        dir_acc, dist_acc = dir_correct / tot, dist_correct / tot
        return dir_acc, dist_acc, (dir_acc + dist_acc) / 2

    def _eval_relations(self, pred_relations: Dict[str, str], gt_room_state_full: BaseRoom, assistant_response: str, pred_json: Dict) -> RelationsCogMapTurnLog:
        # Observed: include only observed names; include initial_pos at agent.init_pos; exclude agent
        # Try to find agent initial pos from any Agent present in the full room state
        agent_obj = next((o for o in gt_room_state_full.objects if isinstance(o, Agent)), None)
        agent_init_pos = agent_obj.init_pos

        # Full: all names; include initial_pos at agent.init_pos; exclude agent
        all_names = {o.name for o in gt_room_state_full.objects if o.name != 'agent'}
        gt_relations_full = room_to_ordered_relations(
            gt_room_state_full,
            include_names=all_names,
            include_initial_pos=True,
            agent_init_pos=agent_init_pos,
        )
        dir_acc_full, dist_acc_full, overall_full = self._relations_accuracies(pred_relations, gt_relations_full)
        return RelationsCogMapTurnLog(
            type="relations",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=None,
            pred_relations=pred_relations,
            gt_relations_full=gt_relations_full,
            metrics=RelationMetrics(dir=float(dir_acc_full), dist=float(dist_acc_full), overall=float(overall_full), valid=True),
        )

    # =============================== Relations helpers/eval ===============================
    def _parse_predicted_relations(self, json_data: Dict[str, Any]) -> Dict[str, str]:
        from ..utils.relation_codes import decode_relation_codes, make_ordered_pair_key, parse_pair_key
        out: Dict[str, str] = {}
        assert isinstance(json_data, dict), f"json_data must be a dict, but got {type(json_data)}"

        candidates = []
        if isinstance(json_data, dict):
            candidates.append(json_data)
        candidates.append(json_data)
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            for k, v in cand.items():
                if not isinstance(k, str):
                    continue
                a, b = parse_pair_key(k)
                if not a or not b:
                    continue
                a, b = a.replace('_', ' '), b.replace('_', ' ')
                key = make_ordered_pair_key(a, b)
                if isinstance(v, str):
                    d, r = decode_relation_codes(v)
                    if d and r:
                        out[key] = f"({d}, {r})"
                elif isinstance(v, dict):
                    d, r = str(v.get('dir', '')).strip().lower(), str(v.get('dist', '')).strip().lower()
                    if d and r:
                        d1, r1 = decode_relation_codes(f"({d},{r})")
                        if d1 and r1:
                            out[key] = f"({d1}, {r1})"
        return out

    def evaluate_cogmaps(self, responses_by_type: Dict[str, str], gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]]) -> CognitiveMapTurnLog:
        """Evaluate multiple types and record one aggregate log for the turn."""
        out = CognitiveMapTurnLog()
        for map_type_key, resp in (responses_by_type or {}).items():
            if not isinstance(resp, str):
                continue
            single = self.evaluate_cogmap_type(resp, gt_room, gt_agent, observed_items, map_type_key)
            setattr(out, f"{single.type}_log", single)
        # Consistency fields per turn
        summary = ConsistencySummary()
        if out.local_log and out.global_log and out.local_log.extraction_success and out.global_log.extraction_success:
            cm = local_vs_global_consistency(
                out.local_log.pred_room_state,
                out.global_log.pred_room_state,
                gt_agent,
                allow_scale=bool(self.config.get('pos_allow_scale', False)),
                pos_norm_L=self._pos_norm_L,
            )
            summary.local_vs_global = cm
        # Rooms vs Global (only when both predicted)
        if out.rooms_log and out.global_log and out.rooms_log.extraction_success and out.global_log.extraction_success:
            avg, per_room = rooms_vs_global_consistency(
                out.rooms_log.pred_rooms_state or {},
                out.global_log.pred_room_state,
                gt_room,
                gt_agent,
                self.entry_gate_by_room,
                allow_scale=bool(self.config.get('pos_allow_scale', False)),
                pos_norm_L=self._pos_norm_L,
            )
            summary.rooms_vs_global_avg = avg
            summary.rooms_vs_global_per_room = per_room
        # Map vs Relations consistency
        if out.global_log and out.relations_log and out.relations_log.extraction_success and out.global_log.extraction_success:
            score = map_vs_relations_consistency(
                out.relations_log.pred_relations or {},
                out.global_log.pred_room_state,
            )
            summary.map_vs_relations = float(score)
        # Relations self-consistency
        if out.relations_log and out.relations_log.extraction_success:
            score_rel = relations_consistency(out.relations_log.pred_relations or {})
            summary.relations_consistency = float(score_rel)
        out.consistency = summary
        return out
            

    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict], exp_type: str = None) -> Dict[str, Any]:
        """Aggregate cognitive map metrics per scenario.

        exp_type in {
            'active': error + consistency + correctness,
            'passive': correctness (global only),
        }
        Prefer precomputed per-sample metrics when available.
        """
        assert isinstance(env_data_list, list) and len(env_data_list) > 0, "env_data_list must be a non-empty list"

        pre_list = [cogmap for s in env_data_list if (cogmap := (s.get('metrics') or {}).get('cogmap')) is not None]
        if exp_type == 'active':
            exploration = avg_nested_dicts([m.get('exploration') or {} for m in pre_list])
            evaluation = avg_nested_dicts([m.get('evaluation') or {} for m in pre_list])
            update_turn = avg_nested_dicts([{'cogmap_update_per_turn': m.get('cogmap_update_per_turn') or {}} for m in pre_list]).get('cogmap_update_per_turn', {})
            full_turn = avg_nested_dicts([{'cogmap_full_per_turn': m.get('cogmap_full_per_turn') or {}} for m in pre_list]).get('cogmap_full_per_turn', {})
            return {
                'exploration': exploration,
                'evaluation': evaluation if evaluation else {'correctness': {}},
                'cogmap_update_per_turn': update_turn,
                'cogmap_full_per_turn': full_turn,
            }
        if exp_type == 'passive':
            exploration = avg_nested_dicts([m.get('exploration') or {} for m in pre_list])
            return {'exploration': {'correctness': {'global_full': (exploration.get('correctness') or {}).get('global_full', {})}}}

        # Default: average nested
        return avg_nested_dicts(pre_list)

    @staticmethod
    def aggregate_per_sample(env_data: Dict[str, Any], exp_type: str | None = None) -> Dict[str, Any]:
        """Aggregate cognitive-map metrics within a single sample.
        Returns exploration error/correctness/consistency and per-turn global metrics.
        """
        # Helper: get exploration turns' cogmap logs
        turn_logs = env_data.get('env_turn_logs') or []
        cog_logs = []
        for t in turn_logs:
            if t.get('is_exploration_phase', False) and t.get('cogmap_log'):
                cog_logs.append(t['cogmap_log'])
        if not cog_logs:
            return {}
        # Use shared helper to find last exploration cogmap
        
        last = get_last_exploration_cogmap(env_data)
        false_belief_metrics = get_false_belief_metrics(env_data)
        # Error: average local/global metrics over turns
        def _avg_maps(dicts: List[Dict[str, Any]], path: List[str]) -> MapCogMetrics:
            mats: List[MapCogMetrics] = []
            for d in dicts:
                cur = d
                ok = True
                for key in path:
                    if isinstance(cur, dict) and key in cur:
                        cur = cur[key]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, dict):
                    m = MapCogMetrics.from_dict(cur)
                    if m.valid:
                        mats.append(m)
            return MapCogMetrics.average(mats) if mats else MapCogMetrics.invalid()

        error = {
            'local_vs_gt_local_avg': _avg_maps(cog_logs, ['local', 'metrics']).to_dict(),
            'global_vs_gt_global_avg': _avg_maps(cog_logs, ['global', 'metrics']).to_dict(),
            'agent_vs_gt_agent_avg': _avg_maps(cog_logs, ['global', 'metric_agent']).to_dict(),
        }

        # Correctness: last global_full and relations_full
        correctness = {
            'last_global_vs_gt_full': (lambda _m: (_m.to_dict() if _m.valid else {}))(MapCogMetrics.from_dict((((last or {}).get('global') or {}).get('metrics_full') or {}))),
            'last_relations_vs_gt_full': (lambda _r: (_r.to_dict() if _r.valid else {}))(RelationMetrics.from_dict((((last or {}).get('relations') or {}).get('metrics') or {}))),
        }

        # Consistency
        # local_vs_global average over turns
        def _avg_consistency_lvsg(dicts: List[Dict[str, Any]]) -> MapCogMetrics:
            mats: List[MapCogMetrics] = []
            for d in dicts:
                cm = (d.get('consistency') or {}).get('local_vs_global') or {}
                m = MapCogMetrics.from_dict(cm)
                if m.valid:
                    mats.append(m)
            return MapCogMetrics.average(mats) if mats else MapCogMetrics.invalid()

        cons_last = (last or {}).get('consistency') or {}
        consistency = {
            'local_vs_global_avg': _avg_consistency_lvsg(cog_logs).to_dict(),
            'rooms_vs_global_last': MapCogMetrics.from_dict(((cons_last.get('rooms_vs_global') or {}).get('average') or {})).to_dict(),
            'map_vs_relations_last': (float(cons_last.get('map_vs_relations')) if isinstance(cons_last.get('map_vs_relations'), (int, float)) else None),
            'relations_consistency_last': (float(cons_last.get('relations_consistency')) if isinstance(cons_last.get('relations_consistency'), (int, float)) else None),
            'stability_avg': MapCogMetrics.average(stability(env_data)).to_dict(),
        }

        # Per-turn global metrics (concise helper)
        per_turn_update, per_turn_full = CognitiveMapManager.compute_per_turn_global_metrics(cog_logs)
        false_belief_acc = BaseCogMetrics.average([BaseCogMetrics.from_dict(m) for m in false_belief_metrics]).to_dict() if false_belief_metrics else None
        if exp_type == 'passive':
            return {
                'exploration': {
                    'correctness': {
                        'global_full': correctness['last_global_vs_gt_full']
                    }
                }
            }

        return {
            'exploration': {
                'error': error,
                'correctness': correctness,
                'consistency': consistency,
            },
            'evaluation': {
                'false_belief_acc': false_belief_acc,
            },
            'cogmap_update_per_turn': per_turn_update,
            'cogmap_full_per_turn': per_turn_full,
        }

    @staticmethod
    def compute_per_turn_global_metrics(cog_logs: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Return (update, full) per-turn global metric lists."""
        per_turn_update = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        per_turn_full = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        for d in cog_logs:
            g = d.get('global') or {}
            mu = MapCogMetrics.from_dict(g.get('metrics') or {})
            mf = MapCogMetrics.from_dict(g.get('metrics_full') or {})
            per_turn_update['dir'].append(float(mu.dir) if mu.valid else None)
            per_turn_update['facing'].append(float(mu.facing) if mu.valid else None)
            per_turn_update['pos'].append(float(mu.pos) if mu.valid else None)
            per_turn_update['overall'].append(float(mu.overall) if mu.valid else None)
            per_turn_full['dir'].append(float(mf.dir) if mf.valid else None)
            per_turn_full['facing'].append(float(mf.facing) if mf.valid else None)
            per_turn_full['pos'].append(float(mf.pos) if mf.valid else None)
            per_turn_full['overall'].append(float(mf.overall) if mf.valid else None)
        return per_turn_update, per_turn_full
    
    # register entry gates for active exploratoin
    def _register_active_entry_gate(self, gt_room) -> None:
        """
        Register entry gates for rooms based on room structure.
        For simplicity, assign the first gate connecting to room 1 as the entry gate for each room.
        Room 1 is considered the starting room and doesn't get an entry gate.
        """
        if not hasattr(gt_room, 'gates') or not gt_room.gates:
            return
            
        # Set room 1 as the starting room
        if self._start_room_id is None:
            self._start_room_id = 1

        # For each room (except room 1), find its connection to room 1 or already processed rooms
        processed_rooms = {self._start_room_id}
        
        # Keep processing until no more rooms can be processed
        changed = True
        while changed:
            changed = False
            for g in gt_room.gates:
                if len(g.room_id) == 2:
                    room_a, room_b = int(g.room_id[0]), int(g.room_id[1])
                    
                    # If one room is processed and the other isn't, register the gate for the unprocessed room
                    if room_a in processed_rooms and room_b not in processed_rooms:
                        if room_b not in self.entry_gate_by_room:
                            self.entry_gate_by_room[room_b] = g.name
                            processed_rooms.add(room_b)
                            changed = True
                    elif room_b in processed_rooms and room_a not in processed_rooms:
                        if room_a not in self.entry_gate_by_room:
                            self.entry_gate_by_room[room_a] = g.name
                            processed_rooms.add(room_a)
                            changed = True

    # =============================== Parsing helpers =============================== 
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON content from text."""
        # Try fenced blocks first
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates = fenced if fenced else []

        # Fallback: scan for outermost balanced braces
        stack, start = [], None
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i+1])
                    start = None

        # Try to load the largest candidate
        candidates.sort(key=len, reverse=True)
        for cand in candidates:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
        return None
    
    def _parse_section_to_baseroom(self, mapping: Dict[str, Any], room_name: str) -> Optional[BaseRoom]:
        """Parse a single section (object_name -> attrs) to BaseRoom.
        Keeps 'agent' as a regular object for evaluation symmetry.
        """
        direction_mapping = {
            "north": np.array([0, 1]),
            "south": np.array([0, -1]),
            "east": np.array([1, 0]),
            "west": np.array([-1, 0])
        }
        objects: List[Object] = []
        for obj_name, obj_info in mapping.items():
            if not isinstance(obj_info, dict):
                continue
            position = obj_info.get('position')
            if not isinstance(position, list) or len(position) != 2 or not all(isinstance(x, (int, float, str)) for x in position):
                continue
            pos = np.array([float(position[0]), float(position[1])])
            facing = obj_info.get('facing', None)
            if isinstance(facing, str):
                ori = direction_mapping.get(facing.lower(), direction_mapping['north'])
                has_orientation = True
            else:
                ori = np.array([0, 0])
                has_orientation = False
            objects.append(Object(name=str(obj_name).replace('_', ' '), pos=pos, ori=ori, has_orientation=has_orientation))

        return BaseRoom(objects=objects, name=room_name)

    def _parse_rooms(self, rooms_sec: Dict[str, Any]) -> Dict[str, BaseRoom]:
        rooms_map: Dict[str, BaseRoom] = {}
        if isinstance(rooms_sec, dict):
            for rid, sec in rooms_sec.items():
                if isinstance(sec, dict):
                    br = self._parse_section_to_baseroom(sec, f"pred_room_{rid}")
                    if br is not None:
                        rooms_map[str(rid)] = br
        return rooms_map

    # =============================== GT constructors =============================== 

    def _baseroom_from_gt(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        objs: List[Object] = []
        # include all non-gate objects
        for o in gt_room.objects:
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=o.has_orientation))
        # include gates
        for g in gt_room.gates:
            objs.append(Object(name=g.name, pos=g.pos.copy(), ori=g.ori.copy(), has_orientation=True))
        # include agent
        objs.append(Agent(name='agent', pos=gt_agent.pos.copy(), ori=gt_agent.ori.copy(), has_orientation=True))
        return BaseRoom(objects=objs, name='gt')

    def _build_gt_global_agent_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        return self._filter_br_by_names(br, {"agent"})
    
    def _build_gt_global_baseroom(self, gt_room: Room, gt_agent: Agent, observed_set: set[str]) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        keep = set(observed_set) | {"agent"}
        return self._filter_br_by_names(br, keep)

    def _build_gt_local_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        visible = self._visible_object_names(gt_room, gt_agent)
        objs: List[Object] = []
        for name in visible:
            o = gt_room.get_object_by_name(name)
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=getattr(o, 'has_orientation', True)))
        raw = BaseRoom(objects=objs, name='gt_local_raw')
        return transform_baseroom(raw, gt_agent.pos, gt_agent.ori)

    def _build_gt_room_baserooms(self, gt_room: Room, gt_agent: Agent, observed_set: set[str]) -> Dict[int, BaseRoom]:
        out: Dict[int, BaseRoom] = {}
        if not isinstance(gt_room, Room):
            return out
        for rid in sorted(gt_room.objects_by_room.keys()):
            gate_name = self.entry_gate_by_room.get(int(rid))
            if gate_name is None: # no entry gate for this room
                gate_name = 'initial'
                anchor_pos = gt_agent.init_pos
                anchor_ori = gt_agent.init_ori
            else:
                gate = next((g for g in gt_room.gates if g.name == gate_name), None)
                anchor_pos, anchor_ori = gate.pos, gate.get_ori_for_room(int(rid))
            # exclude origin gate and agent; include room objects only
            objs: List[Object] = []
            for name in gt_room.objects_by_room[int(rid)]:
                if name == gate_name or name not in observed_set:
                    continue
                o = gt_room.get_object_by_name(name)
                objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=o.has_orientation))
            out[int(rid)] = transform_baseroom(BaseRoom(objects=objs, name=f'gt_room_{rid}'), anchor_pos, anchor_ori)
        return out
    def baseroom_to_json(self, room: BaseRoom, include_gates: bool = True) -> Dict[str, Any]:
        """
        Convert a BaseRoom into a cognitive map–style JSON.

        Args:
            room (BaseRoom): the BaseRoom instance to convert
            include_gates (bool): whether to include gates in the output

        Returns:
            Dict[str, Any]: JSON-like dictionary following the cognitive map schema
        """
        ori_mapping = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}
        out: Dict[str, Any]={}
        # Objects (includes agent if present)
        for obj in room.objects:
            facing = ori_mapping.get(tuple(obj.ori), "")
            out[obj.name] = {
                "position": [int(obj.pos[0]), int(obj.pos[1])],
                "facing": facing
            }

        # Gates
        if include_gates and room.gates:
            for g in room.gates:
                gate_facing = ori_mapping.get(tuple(g.ori), "")
                out[g.name] = {
                    "position": [int(g.pos[0]), int(g.pos[1])],
                    "facing": gate_facing
                }

        return out

    # =============================== Room comparisons =============================== 

    def _compare_baserooms(self, pred_room: BaseRoom, gt_room: BaseRoom) -> MapCogMetrics:
        m = compute_map_metrics(
            pred_room,
            gt_room,
            allow_scale=bool(self.config.get('pos_allow_scale', False)),
            pos_norm_L=self._pos_norm_L,
        )
        return m

    # =============================== Filters and preprocessing =============================== 
    def _filter_br_by_names(self, br: Optional[BaseRoom], keep: set[str]) -> BaseRoom:
        if br is None:
            return BaseRoom(objects=[], name='empty')
        objs = [o for o in br.objects if o.name in keep]
        return BaseRoom(objects=objs, name=br.name)

    def _visible_object_names(self, gt_room: Room, gt_agent: Agent) -> set[str]:
        names = set()
        for o in gt_room.all_objects:
            if BaseAction._is_visible(gt_agent, o):
                names.add(o.name)
        return names

    def _preprocess_predicted(self, json_data: Dict[str, Any], observed: set[str], visible: set[str], gt_room: Room, gt_agent: Agent, map_type: str) -> Dict[str, Any]:
        jd = copy.deepcopy(json_data) if isinstance(json_data, dict) else {}
        gate_names = {g.name for g in gt_room.gates}
        
        def _norm_face_local(f, anchor_ori):
            """For local/room sections - convert relative directions to absolute based on anchor orientation"""
            if not isinstance(f, str):
                return f
            s = f.strip().lower()
            # anchor_ori is like [0,1] for north, [1,0] for east, etc.
            if tuple(anchor_ori) == (0, 1):  # north
                mapping = {"+x": "east", "-x": "west", "+y": "north", "-y": "south"}
            elif tuple(anchor_ori) == (1, 0):  # east
                mapping = {"+x": "south", "-x": "north", "+y": "east", "-y": "west"}
            elif tuple(anchor_ori) == (0, -1):  # south
                mapping = {"+x": "west", "-x": "east", "+y": "south", "-y": "north"}
            elif tuple(anchor_ori) == (-1, 0):  # west
                mapping = {"+x": "north", "-x": "south", "+y": "west", "-y": "east"}
            else:
                # fallback to identity
                return s
            return mapping.get(s, s)

        def _norm_map(obj_map: Dict[str, Any], keep: set = None, anchor_ori = None) -> Dict[str, Any]:
            out = {}
            for name, info in (obj_map or {}).items():
                if not isinstance(info, dict):
                    continue
                # Apply keep filter if provided
                if keep is not None:
                    should_keep, preferred_key = _should_keep_key(name, keep)
                    if not should_keep:
                        continue
                    name = preferred_key
                # normalize facing
                if anchor_ori is not None and "facing" in info:
                    info["facing"] = _norm_face_local(info["facing"], anchor_ori)
                out[name] = info
            return out

        def _should_keep_key(key: str, keep_set: set) -> tuple[bool, str]:
            """Check if key should be kept and return the preferred key name from keep_set"""
            if key in keep_set:
                return True, key
            # Check if key with underscores matches any keep element without underscores
            if '_' in key:
                key_no_underscore = key.replace('_', '')
                for keep_item in keep_set:
                    if keep_item.replace('_', '') == key_no_underscore:
                        return True, keep_item

            return False, key

        def _flatten_nested_json(jd: Dict[str, Any]) -> Dict[str, Any]:
            """Convert nested JSON format (objects/gates arrays) to flat format."""
            if not (isinstance(jd, dict) and ("objects" in jd or "gates" in jd)):
                return jd
                
            flat = {}

            # keep other top-level dict entries (e.g., "agent")
            for k, v in jd.items():
                if k in ("objects", "gates"):
                    continue
                if isinstance(v, dict):
                    flat[str(k).replace('_', ' ')] = v

            def _norm_name(x):
                s = x.get("label") or x.get("name") or x.get("id") or x.get("type") or ""
                s = str(s).strip()
                return s.replace("_", " ") if s else ""

            def _emit(name, info):
                if not name:
                    return
                # require a position-like field
                pos = info.get("position") or info.get("pos") or info.get("xy")
                if pos is None:
                    return
                out = {"position": pos}
                if "facing" in info:
                    out["facing"] = info["facing"]
                # keep other fields, but don't clobber position/facing
                for k, v in info.items():
                    if k not in ("position", "pos", "xy", "facing", "id", "label", "name", "type"):
                        out[k] = v
                flat[name] = out

            def _add(sec):
                if isinstance(sec, list):
                    for it in sec:
                        if isinstance(it, dict):
                            _emit(_norm_name(it), it)
                elif isinstance(sec, dict):
                    for name, info in sec.items():
                        if isinstance(info, dict):
                            _emit(str(name).replace('_', ' '), info)

            _add(jd.get("objects"))
            _add(jd.get("gates"))
            return flat or jd

        # --- Global: keep observed + gates + agent; also handle list-based sections ---
        if map_type == "global":
            # Flatten {"objects":[...], "gates":[...]} into {name: {position, facing, ...}}
            jd = _flatten_nested_json(jd)
            keep = set(observed) | gate_names | {"agent"}
            jd = _norm_map(jd, keep)
            return self._parse_section_to_baseroom(jd, "pred_global") or BaseRoom(objects=[], name="pred_global")

        if map_type == "false_belief":
            # Handle nested format same as global
            jd = _flatten_nested_json(jd)
            # keep all objects
            jd = _norm_map(jd, observed)
            return self._parse_section_to_baseroom(jd, "pred_false_belief") or BaseRoom(objects=[], name="pred_false_belief")
        # --- Local: drop origin + keep only visible objects ---
        if map_type == "local":
            # Handle nested format if present
            jd = _flatten_nested_json(jd)
            if "objects" in jd:
                jd = jd["objects"]
            jd = _norm_map(jd, visible, gt_agent.ori)
            return self._parse_section_to_baseroom(jd, "pred_local") or BaseRoom(objects=[], name="pred_local")

        if map_type == "rooms":
            out_rooms = {}
            for rid, sec in jd.items():
                if not isinstance(sec, dict):
                    continue
                try:
                    rid = int(rid)
                except Exception:
                    print(f"Error parsing room id: {rid}")
                    continue
                # Handle nested format within each room section
                sec = _flatten_nested_json(sec)
                inner = sec.get("objects", sec)  # sometimes wrapped in {"origin":..., "objects":{...}}
                keep = {
                    n
                    for n in observed
                    if n in gt_room.room_by_object
                    and gt_room.room_by_object[n] == int(rid)
                }
                # Get gate orientation for this room
                gate_name = self.entry_gate_by_room.get(int(rid))
                if gate_name:
                    gate = next((g for g in gt_room.gates if g.name == gate_name), None)
                    gate_ori = gate.get_ori_for_room(int(rid))
                    out_rooms[str(rid)] = self._parse_section_to_baseroom(_norm_map(inner, keep, gate_ori), f"pred_room_{rid}")
                elif str(rid) == str(self._start_room_id):
                    # starting room; use agent init_ori
                    out_rooms[str(rid)] = self._parse_section_to_baseroom(_norm_map(inner, keep, gt_agent.init_ori), f"pred_room_{rid}")
                else:
                    # fallback if no entry gate
                    out_rooms[str(rid)] = self._parse_section_to_baseroom(_norm_map(inner, keep), f"pred_room_{rid}")
            return  out_rooms
        raise ValueError(f"Invalid map_type: {map_type}")


    def _ensure_pos_norm_L(self, gt_room: Room, gt_agent: Agent) -> None:
        if self._pos_norm_L is not None:
            return
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        keep = {o.name for o in gt_room.objects}
        br = self._filter_br_by_names(br, keep)
        if not br.objects:
            self._pos_norm_L = 1.0
            return
        P = np.array([o.pos for o in br.objects], dtype=float)
        L = float(np.sqrt((P ** 2).sum(axis=1).mean()))
        self._pos_norm_L = (L if L > 0 else 1.0)

    # =============================== Gates evaluation =============================== 
    def _gt_gate_connections_dict(self, gt_room: Room) -> Dict[str, Any]:
        """Return {gate_name: {'connects':[room_id_a, room_id_b]}} from GT room."""
        out: Dict[str, Any] = {}
        for g in gt_room.gates:
            # expect Gate.room_id like [a,b]
            if isinstance(g.room_id, (list, tuple)) and len(g.room_id) == 2:
                out[g.name] = {"connects": [int(g.room_id[0]), int(g.room_id[1])]}
        return out

    def _evaluate_gate_connections(self, pred_gates: Dict[str, Any], gt_room: Room) -> float:
        if not isinstance(gt_room, Room):
            return 0.0
        gt_gates = self._gt_gate_connections_dict(gt_room)
        if not gt_gates:
            return 0.0
        correct = tot = 0
        for gate_name, gt_info in gt_gates.items():
            gt_conn = sorted([int(x) for x in gt_info.get("connects", [])])
            pred = pred_gates.get(gate_name, {}) if isinstance(pred_gates, dict) else {}
            pred_conn = pred.get("connects", []) if isinstance(pred, dict) else []
            try:
                pred_conn_int = sorted([int(x) for x in pred_conn])
            except Exception:
                pred_conn_int = []
            if gt_conn == pred_conn_int:
                correct += 1
            tot += 1

        return float(correct) / float(tot) if tot > 0 else 0.0


def test_evaluate_cogmaps():
    """Test function to demonstrate calling CognitiveMapManager.evaluate_cogmaps method."""
    import json
    import numpy as np

    # Path to the JSON file
    json_file_path = "/root/VAGEN/results/gpt-5-mini_d7e577c3a5061080/4dd45e3077e7610e/text/active/exploration_turn_logs.json"

    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    turn_log = data[4]

    print(f"Selected turn number: {turn_log.get('turn_number', 'Unknown')}")
    print(f"Total turns available: {len(data)}")

    observed_items = turn_log.get('observed_items', [])
    room_state_data = turn_log.get('room_state', {})
    agent_state_data = turn_log.get('agent_state', {})

    # Prepare responses by type
    # Since cogmap_response is None, let's use the original_response from cogmap_log
    responses_by_type = {}
    cogmap_log = turn_log.get('cogmap_log', {})

    # Extract original responses from each cogmap type
    for map_type in ['global', 'local', 'rooms', 'relations']:
        if map_type in cogmap_log and isinstance(cogmap_log[map_type], dict):
            original_response = cogmap_log[map_type].get('original_response', '')
            if original_response:
                responses_by_type[map_type] = original_response


    print(f"\n📋 Constructing gt_room and gt_agent from turn data...")

    # Import required classes
    from ..core.room import Room
    from ..core.object import Agent

    # Construct gt_room directly from room_state_data using Room.from_dict
    gt_room = Room.from_dict(room_state_data)

    # Construct gt_agent from agent_state_data using Agent.from_dict
    gt_agent = Agent.from_dict(agent_state_data)

    print(f"✅ Constructed gt_room with {len(gt_room.objects)} objects and {len(gt_room.gates)} gates")
    print(f"✅ Constructed gt_agent at position {gt_agent.pos} facing {gt_agent.ori}")

    # Create CognitiveMapManager instance
    manager = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")

    print(f"\n🚀 Calling manager.evaluate_cogmaps()...")

    # Call the actual evaluate_cogmaps method
    result = manager.evaluate_cogmaps(responses_by_type, gt_room, gt_agent, observed_items)

    print(f"✅ Successfully called evaluate_cogmaps!")
    print(f"📊 Result type: {type(result)}")

    # Display the results
    if result:
        print(f"\n📊 New evaluation results:")
        result_dict = result.to_dict()
        for map_type, log_data in result_dict.items():
            if isinstance(log_data, dict) and 'metrics' in log_data:
                metrics = log_data['metrics']
                print(f"   {map_type}: {metrics}")


if __name__ == "__main__":
    # Run the test function
    test_evaluate_cogmaps()