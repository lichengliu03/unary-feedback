"""Forward/Backward navigation tasks with shared helpers.

ForwardFOVEvaluationTask: predict final observation from an action sequence.
BackwardNavEvaluationTask: infer action sequence from a final observation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Agent, Object, Gate
from ..core.relationship import EgoFrontBins, StandardDistanceBins, PairwiseRelationshipDiscrete, OrientationRel
from ..actions import BaseAction, ObserveAction, RotateAction, MoveAction
from ..managers.exploration_manager import ExplorationManager
from ..utils.action_utils import action_results_to_text
from ..utils.utils import hash

"""Small, shared helpers. Use actions (Observe/Rotate/JumpTo) via ExplorationManager."""

def _closest_cardinal(vec: np.ndarray) -> np.ndarray:
    basis = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]
    dots = [float(np.dot(vec, b)) for b in basis]
    return basis[int(np.argmax(dots))]


def _ori_to_name(ori: Tuple[int, int]) -> str:
    mapping = {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}
    return mapping.get(tuple(int(x) for x in ori), "north")


def _ordinal(n: int) -> str:
    return f"{int(n)}th" if n > 3 else "1st" if n == 1 else "2nd" if n == 2 else "3rd" if n == 3 else f"{int(n)}th"


def _nearfar_phrase(i: int, k: int) -> str:
    if i == 1: return "nearest"
    if i == k: return "farthest"
    return f"{_ordinal(i)} nearest"


def _ori_to_deg(ori: Tuple[int, int]) -> int:
    mapping = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}
    return mapping[tuple(int(x) for x in ori)]


class BaseNavEvaluationTask(BaseEvaluationTask):
    """Shared navigation helpers for both tasks."""

    def _agent_from_init(self) -> Agent:
        a = self.agent.copy(); a.pos = self.agent.init_pos.copy(); a.ori = self.agent.init_ori.copy()
        a.room_id = self.agent.init_room_id if getattr(self.agent, 'init_room_id', None) is not None else a.room_id
        if a.room_id is None:
            info = self.room.get_cell_info(int(a.pos[0]), int(a.pos[1])); a.room_id = info.get('room_id', a.room_id)
        return a


    def _move_simple(self, agent: Agent, name: str) -> None:
        obj = self.room.get_object_by_name(name)
        agent.pos = obj.pos.copy()
        agent.room_id = obj.room_id

    def _rotate_to_face(self, mgr: ExplorationManager, target: Object):
        vec = target.pos - mgr.agent.pos
        if np.allclose(vec, 0):
            raise ValueError("Target is at the same position as the agent")
        desired = _closest_cardinal(vec)
        cur, des = _ori_to_deg(tuple(int(x) for x in mgr.agent.ori)), _ori_to_deg(tuple(int(x) for x in desired))
        delta = (des - cur + 540) % 360 - 180
        if int(delta) == 0:
            return None
        return mgr.execute_success_action(RotateAction(int(delta)))

    def _describe_target(self, mgr: ExplorationManager, target: Object) -> str:
        bin_sys, dist_sys = EgoFrontBins(), StandardDistanceBins() # TODO whether use cardinal bins
        rel_t = PairwiseRelationshipDiscrete.relationship(tuple(target.pos), tuple(mgr.agent.pos), anchor_ori=tuple(mgr.agent.ori), bin_system=bin_sys, distance_bin_system=dist_sys)
        dir_label, dist_label = rel_t.direction.bin_label, rel_t.dist.bin_label
        # Build groups via current Observe; indices only when multiple
        dir_group, dist_group = [], []
        for name in mgr.execute_success_action(ObserveAction()).data['visible_objects']:
            o = self.room.get_object_by_name(name)
            rel = PairwiseRelationshipDiscrete.relationship(tuple(o.pos), tuple(mgr.agent.pos), anchor_ori=tuple(mgr.agent.ori), bin_system=bin_sys, distance_bin_system=dist_sys)
            deg = float(rel.direction.degree)
            if int(rel.direction.bin_id) == int(rel_t.direction.bin_id):
                dir_group.append((o, deg))
            if int(rel.dist.bin_id) == int(rel_t.dist.bin_id):
                dval = float(np.linalg.norm(np.array(o.pos) - np.array(mgr.agent.pos)))
                dist_group.append((o, dval))
        # Clear, compact prompt for indices within groups
        dir_phrase, dist_phrase = None, None
        if len(dir_group) > 1 and len(dist_group) > 1:
            if len(dir_group) > 1:
                dir_group.sort(key=lambda x: x[1])
                idx = 1 + next(i for i, (o, _) in enumerate(dir_group) if o.name == target.name)
                dir_phrase = f"{_ordinal(idx)} from left"
            if len(dist_group) > 1:
                dist_group.sort(key=lambda x: x[1])
                i = 1 + next(i for i, (o, _) in enumerate(dist_group) if o.name == target.name)
                dist_phrase = f"{_nearfar_phrase(i, len(dist_group))} one"
        if dir_phrase or dist_phrase:
            parts = [p for p in [dir_phrase, dist_phrase] if p]
            return f"Among objects which are {dir_label}, {dist_label}, you jump to the " + " also ".join(parts) + "."
        return f"Jump to the object at {dir_label}, {dist_label}."

    def build_action_sequence(self, sequence: List[str], final_ori: Tuple[int, int]) -> Tuple[List[List], Agent]:
        """Return per-step ActionResults groups and end agent. Each target yields [Rotate?, Move]. Final group is Rotate if needed."""
        mgr = ExplorationManager(self.room.copy(), self._agent_from_init())
        per_step: List[List] = []
        for name in sequence:
            target = mgr.exploration_room.get_object_by_name(name)
            group: List = []
            rot_res = self._rotate_to_face(mgr, target)
            if rot_res is not None:
                group.append(rot_res)
            # compute description BEFORE moving
            desc = self._describe_target(mgr, target)
            obs_names = set(mgr.execute_success_action(ObserveAction()).data['visible_objects'])
            move_res = mgr.execute_success_action(MoveAction(name), observed_items=obs_names)
            move_res.message = desc
            group.append(move_res)
            per_step.append(group)
        cur, des = _ori_to_deg(tuple(int(x) for x in mgr.agent.ori)), _ori_to_deg(tuple(int(x) for x in final_ori))
        delta = (des - cur + 540) % 360 - 180
        if int(delta) != 0:
            per_step.append([mgr.execute_success_action(RotateAction(int(delta)))])
        return per_step, mgr.agent.copy()

    def action_sequence_to_string(self, per_step: List[List]) -> str:
        """Render numbered lines; always include the final facing rotation."""
        groups = list(per_step)
        return "\n".join(f"{i+1}. {action_results_to_text(group)}" for i, group in enumerate(groups))

    def _current_rooms(self, agent: Agent) -> List[int]:
        rid = getattr(agent, 'room_id', None)
        if isinstance(rid, list):
            return [int(x) for x in rid]
        if rid is None:
            info = self.room.get_cell_info(int(agent.pos[0]), int(agent.pos[1])); rid = info.get('room_id')
        return [int(rid)] if rid is not None else []

    def _candidates_in_rooms(self, rooms: List[int]) -> List[str]:
        names: List[str] = []
        for rid in rooms:
            names.extend(self.room.objects_by_room.get(int(rid), []))
            names.extend(self.room.gates_by_room.get(int(rid), []))
        return list(dict.fromkeys(names))

    def _generate_plan(self, steps: int = 2) -> Tuple[List[str], Tuple[int, int]]:
        """Plan by moving agent directly. Bias crossing rooms via gates and prefer objects in the other room after stepping onto a gate.
        Final orientation guarantees >=1 object in FOV.
        """
        a = self._agent_from_init()
        seq: List[str] = []
        last_was_gate = False
        other_rooms_after_gate: List[int] = []
        for _ in range(int(steps)):
            rooms = self._current_rooms(a)
            cand = [n for n in self._candidates_in_rooms(rooms) if not np.allclose(self.room.get_object_by_name(n).pos, a.pos)]
            if not cand:
                break
            gate_cand = [n for n in cand if isinstance(self.room.get_object_by_name(n), Gate)]
            non_gate = [n for n in cand if n not in gate_cand]
            if last_was_gate:
                # prefer objects in the other room(s) after gate
                objects_in_other_rooms = [n for n in non_gate if self.room.get_object_by_name(n).room_id in other_rooms_after_gate]
                pool = objects_in_other_rooms or non_gate or gate_cand
                name = str(self.np_random.choice(pool))
            else:
                # bias toward gates to encourage crossing
                if gate_cand and int(self.np_random.integers(0, 10)) < 6:  # ~60% pick gate when available
                    name = str(self.np_random.choice(gate_cand))
                else:
                    name = str(self.np_random.choice(non_gate or cand))
            self._move_simple(a, name)
            # update gate context
            if isinstance(self.room.get_object_by_name(name), Gate):
                last_was_gate = True
                gobj = self.room.get_object_by_name(name)
                other_rooms_after_gate = [int(r) for r in list(gobj.room_id) if int(r) not in rooms]
            else:
                last_was_gate = False
                other_rooms_after_gate = []
            seq.append(name)
        # choose final orientation with >=1 visible object
        valid_oris = []
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            tmp = a.copy(); tmp.ori = np.array(ori)
            if ObserveAction().execute(self.room, tmp).data.get('visible_objects', []):
                valid_oris.append(ori)
        final_ori = tuple(valid_oris[int(self.np_random.integers(0, len(valid_oris)))] if valid_oris else (0, 1))
        return seq, final_ori

    def _observe_relationships(self, end_agent: Agent) -> List[str]:
        """All pairwise relationship strings (exclude local) from end agent's perspective."""
        return ObserveAction().execute(self.room, end_agent.copy()).data.get('relationships', [])

    def _observe_text(self, end_agent: Agent, max_items: int = 3) -> str:
        """Shuffle and join up to max_items pairwise lines in one sentence."""
        rels = self._observe_relationships(end_agent)
        self.np_random.shuffle(rels)
        pick = rels[:max(0, min(max_items, len(rels)))]
        return "; ".join(pick)

    def _is_wrong_forward(self, candidate_text: str, correct_all: set[str]) -> bool:
        parts = [p.strip() for p in candidate_text.split(';') if p.strip()]
        return not all(p in correct_all for p in parts)

    def _is_wrong_backward(self, seq: List[str], final_ori: Tuple[int, int], correct_all: set[str]) -> bool:
        """True if executing (seq, final_ori) yields a DIFFERENT set of pairwise relationships."""
        _, end_agent = self.build_action_sequence(seq, final_ori)
        return set(self._observe_relationships(end_agent)) != set(correct_all)

    def _random_forward_obs(self, end_agent: Agent) -> str:
        """Random slight variant: move to a random visible object's pos or randomize orientation, then return pairwise text (no manager)."""
        a = end_agent.copy()
        vis_names = ObserveAction().execute(self.room, a.copy()).data.get('visible_objects', [])
        if vis_names:
            pick = self.room.get_object_by_name(str(self.np_random.choice(vis_names)))
            a.pos, a.room_id = pick.pos.copy(), pick.room_id
        a.ori = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)][int(self.np_random.integers(0, 4))])
        return self._observe_text(a, max_items=3)

    def _perturb_observation_text(self, end_agent: Agent, max_items: int = 3) -> Optional[str]:
        """Perturb selected relationships: shift dir/dist bins and change orientation following direction task pattern."""
        a = end_agent.copy()
        res = ObserveAction().execute(self.room, a)
        triples = [tr for tr in res.data.get('relation_triples', []) if isinstance(tr.relation, PairwiseRelationshipDiscrete)]
        assert triples, "No pairwise relationships found"

        self.np_random.shuffle(triples)
        selected = triples[:max(0, min(max_items, len(triples)))]
        lines = []

        for tr in selected:
            obj = self.room.get_object_by_name(tr.subject)
            dir_labels = tr.relation.direction.bin_system.LABELS
            dist_labels = tr.relation.dist.bin_system.LABELS
            d_idx, s_idx = int(tr.relation.direction.bin_id), int(tr.relation.dist.bin_id)

            # Decide what to change (direction task style shifts)
            change_dir = self.np_random.random() < 0.3   # 50% chance
            change_dist = self.np_random.random() < 0.3  # 50% chance
            change_ori = self.np_random.random() < 0.3   # 50% chance

            if change_dir:
                dk = self.np_random.choice([-2, -1, 1, 2])
                new_d_idx = (d_idx + dk) % len(dir_labels)  # wrap for direction
                # avoid index 0 or -1 (beyond-fov)
                if new_d_idx not in (0, len(dir_labels) - 1):
                    d_idx = new_d_idx

            if change_dist:
                sk = self.np_random.choice([-2, -1, 1, 2])
                s_idx = max(0, min(len(dist_labels) - 1, s_idx + sk))  # clamp for distance

            # Get orientation string
            if isinstance(obj, Gate):
                rid = a.room_id
                if isinstance(rid, (list, tuple)):
                    rid = list(set(a.room_id) & set(obj.room_id))
                    rid = rid[0] if rid else None
                gori = obj.get_ori_for_room(int(rid)) if rid is not None else obj.ori
                opair = OrientationRel.get_relative_orientation(tuple(gori), tuple(a.ori))
                ori_str = OrientationRel.to_string(opair, 'ego', 'orientation', if_gate=True)
            else:
                if change_ori:
                    rand_ori = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.np_random.integers(0, 4)]
                    opair = OrientationRel.get_relative_orientation(tuple(obj.ori), rand_ori)
                else:
                    opair = OrientationRel.get_relative_orientation(tuple(obj.ori), tuple(a.ori))
                ori_str = OrientationRel.to_string(opair, 'ego', 'orientation')

            lines.append(f"{obj.name}: {dir_labels[d_idx]}, {dist_labels[s_idx]}, {ori_str}")

        return "; ".join(lines)

    def _random_backward_action_string(self) -> Tuple[str, List[str], Tuple[int, int]]:
        """Generate a random short sequence and final orientation, then render action string with final face line."""
        seq, final_ori = self._generate_plan(self.steps)
        per_step, _ = self.build_action_sequence(seq, final_ori)
        return self.action_sequence_to_string(per_step), seq, final_ori

class ForwardFOVEvaluationTask(BaseNavEvaluationTask):
    """Predict final observation from an action sequence."""
    ACTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "You will execute an action sequence.\n"
        "Actions:\n{actions}\n\n"
    )
    QUESTION_TEMPLATE = (
        "What will you observe at the end?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        end_agent: Agent = self._ctx['end_agent']
        final_ori: Tuple[int, int] = tuple(self._ctx['final_ori'])
        correct_obs = str(correct_answer)
        wrong_selected: List[str] = []
        # orientation variants (pairwise only)
        wrong_ori: List[str] = []
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if tuple(ori) == tuple(final_ori):
                continue
            a = end_agent.copy(); a.ori = np.array(ori)
            wrong_ori.append(self._observe_text(a, max_items=3))
        # position variants near→far (pairwise only)
        wrong_pos: List[str] = []
        final_pos = end_agent.pos.copy()
        others = [o for o in self.room.all_objects if not np.allclose(o.pos, final_pos)]
        others.sort(key=lambda o: float(np.linalg.norm(o.pos - final_pos)))
        for o in others: # teleport to a random other object sorted by distance
            b = end_agent.copy()
            b.pos, b.room_id = o.pos.copy(), o.room_id
            if int(self.np_random.integers(0, 2)) == 1:
                b.ori = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)][int(self.np_random.integers(0, 4))])
            wrong_pos.append(self._observe_text(b, max_items=3))
        # mutation/perturbation variants
        wrong_mut: List[str] = []
        for _ in range(3):
            s = self._perturb_observation_text(end_agent, max_items=3)
            if s: wrong_mut.append(s)
        correct_all = set(self._observe_relationships(end_agent))
        def valid(s: str) -> bool:
            return bool(s) and self._is_wrong_forward(s, correct_all)
        
        for cat in (wrong_ori, wrong_pos, wrong_mut):
            cands = [s for s in cat if valid(s) and (s != correct_obs) and (s not in wrong_selected)]
            if cands:
                pick = str(self.np_random.choice(cands))
                if pick not in wrong_selected:
                    wrong_selected.append(pick)
        # Fallback: from combined pool, then random generator
        pool = [s for s in (wrong_ori + wrong_pos + wrong_mut) if valid(s) and (s != correct_obs) and (s not in wrong_selected)]
        self.np_random.shuffle(pool)
        for s in pool:
            if len(wrong_selected) >= 3: break
            wrong_selected.append(s)
        while len(wrong_selected) < 3:
            s = self._random_forward_obs(end_agent)
            if s and valid(s) and (s != correct_obs) and (s not in wrong_selected):
                wrong_selected.append(s)
        choices = [correct_obs] + wrong_selected[:3]
        self.np_random.shuffle(choices)
        choices = choices[:4]
        return choices, int(choices.index(correct_obs))

    @retry_generate_question
    def generate_question(self) -> dict:
        self.steps = int(self.config.get('steps', 2))
        seq, final_ori = self._generate_plan(self.steps)
        per_step, end_agent = self.build_action_sequence(seq, final_ori)
        actions_str = self.action_sequence_to_string(per_step)
        correct_obs = self._observe_text(end_agent, max_items=3)
        self._ctx = {'end_agent': end_agent.copy(), 'final_ori': tuple(final_ori)}
        choices, correct_idx = self.generate_choices(correct_obs)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        self.eval_data.action = self.ACTION_TEMPLATE.format(actions=actions_str)
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(choices_text=choices_text)
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

class BackwardNavEvaluationTask(ForwardFOVEvaluationTask):
    """Infer action sequence from final observation."""
    ACTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "Then you have executed an action sequence and changed to a new location and facing direction.\n"
        "You see the final observation below:\n"
        "{final_obs}\n\n"
    )
    QUESTION_TEMPLATE = (
        "Which action sequence led to this final view?\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def _wrong_by_orientation(self, seq: List[str], final_ori: Tuple[int, int], avoid: Tuple[int, int]) -> Optional[Tuple[str, List[str], Tuple[int, int]]]:
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if tuple(ori) == tuple(avoid): continue
            per_step, _ = self.build_action_sequence(seq, ori)
            return self.action_sequence_to_string(per_step), list(seq), tuple(ori)
        return None

    def _wrong_by_final_object(self, seq: List[str], final_ori: Tuple[int, int]) -> Optional[Tuple[str, List[str], Tuple[int, int]]]:
        if not seq: return None
        to_set = lambda v: set(v) if isinstance(v, list) else {v}
        pool = [o.name for o in self.room.all_objects
                if (
                    o.name != seq[-1]
                    and not np.allclose(o.pos, self.room.get_object_by_name(seq[-2]).pos)
                    and not np.allclose(o.pos, self.room.get_object_by_name(seq[-1]).pos)
                    and to_set(o.room_id).intersection(to_set(self.room.get_object_by_name(seq[-2]).room_id))
                )
            ]
        self.np_random.shuffle(pool)
        for n in pool:
            alt = list(seq[:-1] + [n])
            per_step, _ = self.build_action_sequence(alt, final_ori)
            return self.action_sequence_to_string(per_step), alt, tuple(final_ori)
        return None

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        seq, final_ori = list(self._ctx['seq']), tuple(self._ctx['final_ori'])
        correct = str(correct_answer)
        # compute the correct set of pairwise relationships once
        correct_all = set(self._observe_relationships(self.build_action_sequence(seq, final_ori)[1]))

        wrong_selected: List[str] = []

        def valid_choice(cand: Optional[Tuple[str, List[str], Tuple[int, int]]]) -> Optional[str]:
            if not cand:
                return None
            txt, s, o = cand
            return txt if self._is_wrong_backward(s, o, correct_all) else None

        for cand in (
            self._wrong_by_orientation(seq, final_ori, avoid=final_ori),
            self._wrong_by_final_object(seq, final_ori),
            self._random_backward_action_string(),
        ):
            txt = valid_choice(cand)
            if txt and (txt != correct) and (txt not in wrong_selected):
                wrong_selected.append(txt)

        while len(wrong_selected) < 3:
            txt = valid_choice(self._random_backward_action_string())
            if txt and (txt != correct) and (txt not in wrong_selected):
                wrong_selected.append(txt)

        choices = [correct] + wrong_selected[:3]
        self.np_random.shuffle(choices)
        return choices, int(choices.index(correct))

    @retry_generate_question
    def generate_question(self) -> dict:
        self.steps = int(self.config.get('max_steps', 2))
        seq, final_ori = self._generate_plan(self.steps)
        per_step, end_agent = self.build_action_sequence(seq, final_ori)
        final_obs = action_results_to_text([ObserveAction().execute(self.room, end_agent.copy())])
        correct_actions = self.action_sequence_to_string(per_step)
        self._ctx = {'seq': list(seq), 'final_ori': tuple(final_ori)}
        choices, correct_idx = self.generate_choices(correct_actions)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        self.eval_data.action = self.ACTION_TEMPLATE.format(final_obs=final_obs)
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(choices_text=choices_text)
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question


