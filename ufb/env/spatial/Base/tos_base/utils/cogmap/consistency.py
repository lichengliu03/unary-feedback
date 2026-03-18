from typing import Dict, Tuple, List
import math
import numpy as np
from itertools import combinations
import copy

from ...core.room import BaseRoom, Room, Object
from ...core.object import Agent 
from ...core.relationship import RelationTriple, CardinalBinsAllo, StandardDistanceBins
from ...managers.spatial_solver import SpatialSolver
from .transforms import br_from_anchor_to_initial
from .metrics import compute_map_metrics
from .types import MapCogMetrics
from ..relation_codes import (
    decode_relation_codes, encode_relation_codes, discrete_relation_from_codes,
    make_ordered_pair_key, parse_pair_key, invert_relation_codes_str, invert_pair_key)
from ..relationship_utils import room_to_ordered_relations
from .transforms import transform_baseroom


def compare_on_common_subset(a: BaseRoom | None, b: BaseRoom | None, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if a is None or b is None:
        return MapCogMetrics.invalid()
    names_a = {o.name for o in a.objects}
    if not names_a:
        return MapCogMetrics(dir=1.0, facing=1.0, overall=1.0, pos=1.0)  # No objects in A, trivially perfect
    names_b = {o.name for o in b.objects}
    names = names_a & names_b
    if not names:
        # No overlap -> treat as wrong
        return MapCogMetrics(dir=0.0, facing=0.0, overall=0.0, pos=0.0, valid=True)
    a_sub = BaseRoom(objects=[o for o in a.objects if o.name in names], name=a.name)
    b_sub = BaseRoom(objects=[o for o in b.objects if o.name in names], name=b.name)
    return compute_map_metrics(a_sub, b_sub, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def local_vs_global_consistency(pred_local: BaseRoom | None, pred_global: BaseRoom | None, agent: Agent, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if pred_local is None or pred_global is None:
        return MapCogMetrics.invalid()
    
    # Find predicted agent in global map
    global_agent = next((o for o in pred_global.objects if o.name == 'agent'), None)
    if global_agent is None:
        return MapCogMetrics.invalid()
    
    # Transform global map to use predicted agent as origin (make copy to avoid modifying original)
    
    global_copy = copy.deepcopy(pred_global)
    global_agent_centered = transform_baseroom(global_copy, global_agent.pos, global_agent.ori)
    
    # Compare directly (local should already be agent-centered)
    return compare_on_common_subset(pred_local, global_agent_centered, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def rooms_vs_global_consistency(pred_rooms: Dict[str, BaseRoom], pred_global: BaseRoom | None, room: Room, agent: Agent, entry_gate_by_room: Dict[int, str], allow_scale: bool, pos_norm_L: float | None) -> Tuple[MapCogMetrics, Dict[str, MapCogMetrics]]:
    if pred_global is None:
        return MapCogMetrics.invalid(), {}
    per_room: Dict[str, MapCogMetrics] = {}
    vals: List[MapCogMetrics] = []
    # Iterate over all GT rooms to ensure missing predictions count as 0
    for rid_int in sorted(room.objects_by_room.keys()):
        rid = str(rid_int)
        room_br = pred_rooms.get(rid)
        gate_name = entry_gate_by_room.get(rid_int)
        if gate_name:
            g = next((gg for gg in room.gates if gg.name == gate_name), None)
            if g is None:
                m = MapCogMetrics(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=True)
                per_room[rid] = m
                vals.append(m)
                continue
            gate_pos = g.pos
            gate_ori = g.get_ori_for_room(rid_int)
        elif rid_int == 1:
            gate_pos = agent.init_pos
            gate_ori = agent.init_ori
        else:
            # No anchor info; treat as wrong
            m = MapCogMetrics(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=True)
            per_room[rid] = m
            vals.append(m)
            continue
        if room_br is None:
            m = MapCogMetrics(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=True)
            per_room[rid] = m
            vals.append(m)
            continue
        room_in_initial = br_from_anchor_to_initial(room_br, gate_pos, gate_ori, agent)
        m = compare_on_common_subset(room_in_initial, pred_global, allow_scale=allow_scale, pos_norm_L=pos_norm_L)
        # If invalid comparison (no overlap), count as 0 instead of skipping
        if not m.valid:
            m = MapCogMetrics(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=True)
        per_room[rid] = m
        vals.append(m)
    avg = MapCogMetrics.average(vals) if vals else MapCogMetrics.invalid()
    return avg, per_room


def map_vs_relations_consistency(pred_relations: Dict, pred_global: BaseRoom | None) -> float:
    """Compare consistency between global map coordinates and pairwise relations.

    Args:
        pred_relations: Dict with pairwise relations like {"A|B": "(dir_code, dist_code)"}
        pred_global: BaseRoom with object positions, or None

    Returns:
        Float between 0.0 and 1.0 representing consistency (1.0 = perfect match)
    """
    if not pred_relations or pred_global is None:
        return 0.0

    # Expected relations from map (ordered A|B)
    expected_relations = room_to_ordered_relations(pred_global)

    # Compare with predicted relations
    if not expected_relations:
        return 0.0

    matches = 0
    total = len(expected_relations)

    from ..relation_codes import invert_pair_key
    for pair_key, expected_rel in expected_relations.items():
        # Accept exact order; if opposite provided, invert before compare
        predicted_rel = pred_relations.get(pair_key)
        if predicted_rel is None:
            inv_key = invert_pair_key(pair_key)
            inv_val = pred_relations.get(inv_key)
            if inv_val is not None:
                predicted_rel = invert_relation_codes_str(inv_val)
        if predicted_rel == expected_rel:
            matches += 1
        else:
            # Check partial matches (direction or distance)
            exp_dir, exp_dist = decode_relation_codes(expected_rel)
            pred_dir, pred_dist = decode_relation_codes(predicted_rel)
            if exp_dir == pred_dir or exp_dist == pred_dist:
                matches += 0.5  # Partial credit for partial match

    return min(1.0, matches / total) if total > 0 else 0.0


def relations_consistency(pred_relations: Dict) -> float:
    """Check consistency of pairwise relations using spatial constraint solving.

    For each triple (A, B, C), set A at (0,0), add constraints A-B and A-C,
    then check if the implied B-C relation matches the given B-C relation.

    Args:
        pred_relations: Dict with pairwise relations like {"A|B": "(dir_code, dist_code)"}

    Returns:
        Float between 0.0 and 1.0 representing consistency (1.0 = fully consistent)
    """
    if not pred_relations or len(pred_relations) < 3:
        # Empty or too few relations after successful extraction -> treat as wrong
        return 0.0

    # Extract unique object names (ordered keys preferred)
    all_names_set = set()
    for k in pred_relations.keys():
        a, b = parse_pair_key(k)
        if a and b:
            all_names_set.add(a); all_names_set.add(b)
    all_names = list(all_names_set)
    if len(all_names) < 3:
        return 1.0  # Need at least 3 objects for triangular consistency

    total_checks = 0
    consistent_checks = 0

    # Generate non-repetitive triples

    for name_a, name_b, name_c in combinations(all_names, 3):
        ab_key = make_ordered_pair_key(name_a, name_b); ba_key = invert_pair_key(ab_key)
        ac_key = make_ordered_pair_key(name_a, name_c); ca_key = invert_pair_key(ac_key)
        bc_key = make_ordered_pair_key(name_b, name_c); cb_key = invert_pair_key(bc_key)

        ab_rel = pred_relations.get(ab_key)
        if ab_rel is None and (val := pred_relations.get(ba_key)):
            ab_rel = invert_relation_codes_str(val)
        ac_rel = pred_relations.get(ac_key)
        if ac_rel is None and (val := pred_relations.get(ca_key)):
            ac_rel = invert_relation_codes_str(val)
        bc_rel = pred_relations.get(bc_key)
        if bc_rel is None and (val := pred_relations.get(cb_key)):
            bc_rel = invert_relation_codes_str(val)

        if ab_rel and ac_rel and bc_rel:
            total_checks += 1
            if _check_triple_consistency(name_a, name_b, name_c, ab_rel, ac_rel, bc_rel):
                consistent_checks += 1

    return consistent_checks / total_checks if total_checks > 0 else 1.0


def _check_triple_consistency(name_a: str, name_b: str, name_c: str,
                             ab_rel: str, ac_rel: str, bc_rel: str) -> bool:
    """Check if three pairwise relations form a consistent triangle.

    Uses spatial solver with AC and BC relations to get possible positions for A and B,
    then checks if any derived AB relation matches the given AB relation.
    """
    # Create spatial solver with the three objects
    solver = SpatialSolver([name_a, name_b, name_c], grid_size=20)

    # Set C at origin (easier to reason about AC and BC relations)
    solver.set_initial_position(name_c, (0, 0))

    # Parse discrete relations directly from codes
    ac_dir, ac_dist = decode_relation_codes(ac_rel)
    bc_dir, bc_dist = decode_relation_codes(bc_rel)

    # Build discrete relations from bins only (no grid distance checks)
    ac_discrete_rel = discrete_relation_from_codes(ac_dir, ac_dist)
    bc_discrete_rel = discrete_relation_from_codes(bc_dir, bc_dist)

    # Add constraints: A relative to C, B relative to C
    relation_triples = [
        RelationTriple(subject=name_a, anchor=name_c, relation=ac_discrete_rel, orientation=(0, 1)),
        RelationTriple(subject=name_b, anchor=name_c, relation=bc_discrete_rel, orientation=(0, 1))
    ]

    ok = solver.add_observation(relation_triples)
    if not ok:
        return False

    # Use possible relations instead of enumerating positions
    rel_sets = solver.get_possible_relations(
        max_samples_per_var=50,
        perspective=(0, 1),
        bin_system=CardinalBinsAllo(),
        distance_bin_system=StandardDistanceBins(),
        path_consistent=False,
    )

    # Normalize keys to ordered lookup and code strings
    ab_dir, ab_dist = decode_relation_codes(ab_rel)
    target = encode_relation_codes(ab_dir, ab_dist)
    # rel_sets uses unordered tuple keys (a,b). Check both and invert when needed
    pair = (min(name_a, name_b), max(name_a, name_b))
    s = rel_sets.get(pair, set())
    if not s:
        return False

    # Ensure we compare as A relative to B (unified branch)
    need_invert = not (name_a <= name_b)
    for cand in s:
        d, r = decode_relation_codes(cand)
        code = encode_relation_codes(d, r)
        if need_invert:
            code = invert_relation_codes_str(code)
        if code == target:
            return True
    return False


def stability(env_data_or_logs: Dict | List[Dict], threshold: int = 5,
              allow_scale: bool = False, pos_norm_L: float | None = None) -> List[MapCogMetrics]:
    """Per-adjacent-turn stability using predicted vs GT global maps on unchanged objects.

    For each adjacent exploration turn (t-1 -> t):
    - Select objects with small domain-size change based on possible_positions between t-1 and t
    - Compare current predicted global map vs current GT global (restricted to selected objects)

    Returns a list of MapCogMetrics (one per adjacent pair). Invalid metric for missing data.
    """
    # Normalize input to a list of exploration turns
    if isinstance(env_data_or_logs, dict):
        logs = env_data_or_logs.get('env_turn_logs', []) or []
    else:
        logs = env_data_or_logs or []

    expl = [t for t in logs if t.get('is_exploration_phase')]
    out: List[MapCogMetrics] = []
    if len(expl) <= 1:
        return out

    def _filter_room(br: BaseRoom, keep: set[str]) -> BaseRoom:
        objs = [o for o in br.objects if o.name in keep]
        return BaseRoom(objects=objs, name=br.name)

    for i in range(1, len(expl)):
        prev_log = expl[i - 1]
        curr_log = expl[i]
        prev_pp: Dict[str, List[List[int]]] = (prev_log.get('exploration_log') or {}).get('possible_positions') or {}
        curr_pp: Dict[str, List[List[int]]] = (curr_log.get('exploration_log') or {}).get('possible_positions') or {}

        # Need current predicted and GT global rooms
        g_curr = ((curr_log.get('cogmap_log') or {}).get('global') or {})
        pred_curr = BaseRoom.from_dict((g_curr.get('pred_room_state')) or {})
        gt_curr = BaseRoom.from_dict((g_curr.get('gt_room_state_full') or g_curr.get('gt_room_state')) or {})

        if not prev_pp or not curr_pp or pred_curr is None or gt_curr is None:
            out.append(MapCogMetrics.invalid())
            continue

        # Select unchanged objects based on domain-size change
        selected: set[str] = set()
        for name, prev_pts in prev_pp.items():
            if name in curr_pp and abs(len(prev_pts) - len(curr_pp[name])) < int(threshold):
                selected.add(name)
        if not selected:
            out.append(MapCogMetrics.invalid())
            continue

        # Restrict rooms to selected objects and compare
        pred_sel = _filter_room(pred_curr, selected)
        gt_sel = _filter_room(gt_curr, selected)
        out.append(compare_on_common_subset(pred_sel, gt_sel, allow_scale=allow_scale, pos_norm_L=pos_norm_L))

    return out


__all__ = [
    "compare_on_common_subset",
    "local_vs_global_consistency",
    "rooms_vs_global_consistency",
    "map_vs_relations_consistency",
    "relations_consistency",
    "stability",
]



if __name__ == "__main__":
    print("Testing consistency functions...")

    # Test 1: compare_on_common_subset
    print("\n1. Testing compare_on_common_subset:")
    try:
        # Create test BaseRooms with common objects
        obj1_a = Object(name="chair", pos=[1, 2])
        obj2_a = Object(name="table", pos=[3, 4])
        room_a = BaseRoom(objects=[obj1_a, obj2_a], name="room_a")

        obj1_b = Object(name="chair", pos=[1.1, 2.1])  # Slightly different position
        obj2_b = Object(name="table", pos=[3.2, 4.1])
        room_b = BaseRoom(objects=[obj1_b, obj2_b], name="room_b")

        metrics = compare_on_common_subset(room_a, room_b, allow_scale=False, pos_norm_L=None)
        print(f"Metrics: overall={metrics.overall:.3f}, pos={metrics.pos:.3f}, valid={metrics.valid}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: local_vs_global_consistency
    print("\n2. Testing local_vs_global_consistency:")
    try:
        # Create local and global rooms
        local_obj = Object(name="chair", pos=[0, 1])  # Relative to agent
        pred_local = BaseRoom(objects=[local_obj], name="local")

        global_obj = Object(name="chair", pos=[2, 3])  # Global position
        pred_global = BaseRoom(objects=[global_obj], name="global")

        agent = Agent(pos=[2, 2], ori=[0, 1])  # Agent at (2,2) facing north

        metrics = local_vs_global_consistency(pred_local, pred_global, agent, allow_scale=False, pos_norm_L=None)
        print(f"Metrics: overall={metrics.overall:.3f}, pos={metrics.pos:.3f}, valid={metrics.valid}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: map_vs_relations_consistency
    print("\n3. Testing map_vs_relations_consistency:")
    try:
        # Create BaseRoom with objects
        obj_a = Object(name="A", pos=[0, 0])
        obj_b = Object(name="B", pos=[1, 0])  # B is east of A
        obj_c = Object(name="C", pos=[0, 1])  # C is north of A
        pred_global = BaseRoom(objects=[obj_a, obj_b, obj_c], name="global")

        # Relations that should partially match the map
        # Note: "A|B" means A relative to B (A is relative to B)
        pred_relations = {
            "A|B": "(W, near)",     # A is west of B (since B is at [1,0] and A is at [0,0])
            "A|C": "(S, near)",     # A is south of C (since C is at [0,1] and A is at [0,0])
            "B|C": "(E, mid)"       # B is east of C (since B is at [1,0] and C is at [0,1])
        }

        score = map_vs_relations_consistency(pred_relations, pred_global)
        print(f"Map vs Relations consistency score: {score:.3f}")

        # Test with empty inputs
        score_empty = map_vs_relations_consistency({}, None)
        print(f"Empty inputs score: {score_empty:.3f}")

    except Exception as e:
        print(f"Error: {e}")

    # Test 4: relations_consistency
    print("\n4. Testing relations_consistency:")
    try:
        # Test with simple consistent relations (just 2 pairs)
        # Note: "A|B" means A relative to B
        simple_relations = {
            "A|B": "(W, near)",    # A is west of B
            "A|C": "(S, near)",    # A is south of C
        }

        # Test with triangle relations
        # Assume positions: A=[0,0], B=[1,0], C=[0,1]
        triangle_relations = {
            "A|B": "(W, near)",    # A is west of B (A=[0,0] relative to B=[1,0])
            "A|C": "(S, near)",    # A is south of C (A=[0,0] relative to C=[0,1])
            "B|C": "(SE, near)"     # B is east of C (B=[1,0] relative to C=[0,1])
        }

        # Test with inconsistent relations
        inconsistent_relations = {
            "A|B": "(W, near)",    # A is west of B (correct)
            "A|C": "(S, near)",    # A is south of C (correct)
            "B|C": "(W, far)"      # B is west of C (inconsistent - should be east)
        }

        score1 = relations_consistency(simple_relations)
        print(f"Simple relations (2 pairs) consistency: {score1:.3f}")

        score2 = relations_consistency(triangle_relations)
        print(f"Triangle relations consistency: {score2:.3f}")

        score3 = relations_consistency(inconsistent_relations)
        print(f"Inconsistent relations consistency: {score3:.3f}")

        # Test with empty/insufficient data
        score_empty = relations_consistency({})
        print(f"Empty relations consistency: {score_empty:.3f}")

        score_insufficient = relations_consistency({"A|B": "(E, near)"})
        print(f"Insufficient relations consistency: {score_insufficient:.3f}")

    except Exception as e:
        print(f"Error: {e}")

    print("\nConsistency function tests completed!")


