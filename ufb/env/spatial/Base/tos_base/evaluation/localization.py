"""Localization task: infer your 2D coordinate from a new view."""

from typing import List, Tuple, Any
import numpy as np

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object, Gate
from ..core.relationship import  PairwiseRelationshipDiscrete,  OrientationRel
from ..actions import BaseAction
from ..actions import ObserveAction
from ..utils.utils import hash
def _ori_to_name(ori: Tuple[int, int]) -> str:
    mapping = {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}
    return mapping.get(tuple(int(x) for x in ori), "north")

class BaseLocEvaluationTask(BaseEvaluationTask):
    """Base class for localization tasks."""
    def _pick_room(self) -> int:
        rids = [int(r) for r in self.room.objects_by_room.keys() if isinstance(r, int) and r > 0]
        self.np_random.shuffle(rids)
        for rid in rids:
            names = self.room.objects_by_room.get(int(rid), [])
            if len(names) < 2:
                continue
            objs = [self.room.get_object_by_name(n) for n in names]
            ok = False
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    d = float(np.linalg.norm(objs[i].pos - objs[j].pos))
                    if d > 1.0 + 1e-6:
                        ok = True; break
                if ok: break
            if ok:
                return int(rid)
        return 1

    def _sample_valid_agent_pose(self) -> Tuple[Tuple[int, int], Tuple[int, int], int, List[Object], List[Object]]:
        """Pick a room and a pose: >=1 visible object and >=1 hidden object in that room."""
        rid = self._pick_room()
        xmin, xmax, ymin, ymax = self.room.get_boundary(room_id=rid)
        coords = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)]
        self.np_random.shuffle(coords)
        for pos in coords:
            if self.room.get_cell_info(pos[0], pos[1])['object_name']:
                continue
            for ori in self.np_random.permutation([(0,1), (1,0), (0,-1), (-1,0)]):
                tmp = self.agent.copy()
                tmp.pos, tmp.ori, tmp.room_id = np.array(pos), np.array(ori), rid
                in_room = [o for o in self.room.objects if int(o.room_id) == rid]
                vis = [o for o in in_room if BaseAction._is_visible(tmp, o) and not np.allclose(o.pos, tmp.pos)]
                hid = [o for o in in_room if o not in vis and not np.allclose(o.pos, tmp.pos)]
                # Filter to objects within distance 5 of tmp position
                nearby_hid = [o for o in hid if np.linalg.norm(o.pos - tmp.pos) <= 5]
                if nearby_hid:
                    hid = nearby_hid
                if len(vis) >= 1 and hid:
                    return pos, ori, rid, vis, hid
        raise ValueError("No valid pose found")


class BackwardLocEvaluationTask(BaseLocEvaluationTask):
    """Localize your own coordinate (x, y) and orientation."""
    ACTION_TEMPLATE = (
        "You change to a new location and facing direction\n"
        "{observations}\n"
    )
    QUESTION_TEMPLATE = (
        "Treat {origin_name} as the origin (0, 0), and your starting facing direction is north.\n"
        "What is your current 2D coordinate (x, y) and facing direction?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, visible_objs, hidden_objs = self._sample_valid_agent_pose()
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)
        origin_obj = self.np_random.choice(hidden_objs)
        observations = self._take_observations()

        # correct answer: your coord relative to origin and orientation
        origin_pos = tuple(origin_obj.pos)
        correct_coord = (int(self.agent.pos[0]) - origin_pos[0], int(self.agent.pos[1]) - origin_pos[1])
        correct_orientation = _ori_to_name(tuple(self.agent.ori))
        correct_answer = (correct_coord, correct_orientation)
        # store ctx for choices
        self._ctx = {
            'rid': int(rid),
            'origin_pos': origin_pos,
            'visible_names': [o.name for o in visible_objs],
            'agent_ori': tuple(self.agent.ori),
        }
        choices, correct_idx = self.generate_choices(correct_answer)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        self.eval_data.action = self.ACTION_TEMPLATE.format(
            observations=observations
        )
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(
            origin_name=origin_obj.name,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    def generate_choices(self, correct_answer: Tuple[Tuple[int, int], str]) -> Tuple[List[str], int]:
        rid = int(self._ctx['rid'])
        origin_pos = tuple(self._ctx['origin_pos'])
        agent_ori = tuple(self._ctx['agent_ori'])
        visible_names = list(self._ctx['visible_names'])

        correct_coord, correct_ori_name = correct_answer
        
        # true discrete rels: object -> agent (from agent orientation)
        true_rels = {}
        for name in visible_names:
            obj = self.room.get_object_by_name(name)
            rel = PairwiseRelationshipDiscrete.relationship(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=agent_ori)
            true_rels[name] = (int(rel.direction.bin_id), int(rel.dist.bin_id))

        def fmt_choice(coord: Tuple[int, int], ori_name: str) -> str:
            return f"({int(coord[0])}, {int(coord[1])}) facing {ori_name}"

        correct_text = fmt_choice(correct_coord, correct_ori_name)
        out, seen = [correct_text], {correct_text}
        
        orientations = ["north", "east", "south", "west"]
        ori_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Random candidate selection with shuffled coordinates and orientations
        xmin, xmax, ymin, ymax = self.room.get_boundary(room_id=rid)
        candidates = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)]
        self.np_random.shuffle(candidates)
        self.np_random.shuffle(orientations)  # Also shuffle orientations
        
        used_coords = set()  # Track used coordinates to ensure diversity
        
        for x, y in candidates:
            if len(out) >= 4:
                break
            if (x, y) == tuple(self.agent.pos) or np.linalg.norm(np.array((x, y)) - np.array(self.agent.pos)) < 2:
                continue
            if self.room.get_cell_info(x, y)['object_name']:
                continue
                
            wrong_coord = (int(x - origin_pos[0]), int(y - origin_pos[1]))
            
            # Skip if we already used this coordinate
            if wrong_coord in used_coords or wrong_coord == correct_coord:
                continue
            
            # Pick a random orientation for this position
            ori_idx = self.np_random.integers(0, 4)
            ori_vec, ori_name = ori_vectors[ori_idx], orientations[ori_idx]
            
            # Skip if this matches the correct answer
            if wrong_coord == correct_coord and ori_name == correct_ori_name:
                continue
            
            # Check if relationships would mismatch with this wrong pose (inconsistent observations)
            mismatch = False
            for name in visible_names:
                obj = self.room.get_object_by_name(name)
                rel = PairwiseRelationshipDiscrete.relationship(tuple(obj.pos), (int(x), int(y)), anchor_ori=ori_vec)
                pair = (int(rel.direction.bin_id), int(rel.dist.bin_id))
                if pair != true_rels[name]:
                    mismatch = True
                    break
            
            # Only add if observations would be inconsistent (wrong choice)
            if mismatch:
                choice_text = fmt_choice(wrong_coord, ori_name)
                if choice_text not in seen:
                    out.append(choice_text); seen.add(choice_text)
                    used_coords.add(wrong_coord)  # Mark this coordinate as used

        # If still not enough choices, add some with correct position but wrong orientation
        if len(out) < 4:
            for ori_name in orientations:
                if len(out) >= 4:
                    break
                if ori_name != correct_ori_name:
                    choice_text = fmt_choice(correct_coord, ori_name)
                    if choice_text not in seen:
                        out.append(choice_text); seen.add(choice_text)

        self.np_random.shuffle(out)
        return out, out.index(correct_text)


class ForwardLocEvaluationTask(BaseLocEvaluationTask):
    ACTION_TEMPLATE = (
        "Treat {origin_name} as the origin (0, 0), and your starting facing direction is north.\n"
        "You move to {loc} and face {direction}.\n"
    )
    QUESTION_TEMPLATE = (
        "What will you observe?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def _observe_relationships(self, end_agent) -> List[str]:
        return ObserveAction().execute(self.room, end_agent.copy(), free_position=True).data.get('relationships', [])

    def _observe_text(self, end_agent, max_items: int = 3) -> str:
        rels = self._observe_relationships(end_agent)
        self.np_random.shuffle(rels)
        pick = rels[:max(0, min(max_items, len(rels)))]
        return "; ".join(pick)

    def _is_wrong_forward(self, candidate_text: str, correct_all: set[str]) -> bool:
        parts = [p.strip() for p in candidate_text.split(';') if p.strip()]
        return not all(p in correct_all for p in parts)

    def _perturb_observation_text(self, end_agent, max_items: int = 3) -> str | None:
        a = end_agent.copy()
        res = ObserveAction().execute(self.room, a, free_position=True)
        triples = [tr for tr in res.data.get('relation_triples', []) if isinstance(tr.relation, PairwiseRelationshipDiscrete)]
        if not triples:
            return None
        self.np_random.shuffle(triples)
        selected = triples[:max(0, min(max_items, len(triples)))]
        lines: List[str] = []
        for tr in selected:
            obj = self.room.get_object_by_name(tr.subject)
            dir_labels = tr.relation.direction.bin_system.LABELS
            dist_labels = tr.relation.dist.bin_system.LABELS
            d_idx, s_idx = int(tr.relation.direction.bin_id), int(tr.relation.dist.bin_id)
            change_dir = self.np_random.random() < 0.3
            change_dist = self.np_random.random() < 0.3
            change_ori = self.np_random.random() < 0.3
            if change_dir:
                dk = self.np_random.choice([-2, -1, 1, 2])
                new_d_idx = (d_idx + dk) % len(dir_labels)
                if new_d_idx not in (0, len(dir_labels) - 1):
                    d_idx = new_d_idx
            if change_dist:
                sk = self.np_random.choice([-2, -1, 1, 2])
                s_idx = max(0, min(len(dist_labels) - 1, s_idx + sk))
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

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, _, hidden_objs = self._sample_valid_agent_pose()
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)
        origin_obj = self.np_random.choice(hidden_objs)

        # question fields
        origin_pos = tuple(origin_obj.pos)
        loc_rel = (int(self.agent.pos[0]) - origin_pos[0], int(self.agent.pos[1]) - origin_pos[1])
        dir_name = _ori_to_name(tuple(self.agent.ori))

        # compute correct observation text (pairwise-only, compact)
        correct_obs = self._observe_text(self.agent, max_items=3)
        # store context including origin name for filtering choices later
        self._ctx = {
            'end_agent': self.agent.copy(),
            'final_ori': tuple(self.agent.ori),
            'origin_name': origin_obj.name,
        }

        # build choices
        choices, correct_idx = self.generate_choices(correct_obs)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.action = self.ACTION_TEMPLATE.format(
            origin_name=origin_obj.name,
            loc=f"({int(loc_rel[0])}, {int(loc_rel[1])})",
            direction=dir_name,
        )
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        end_agent = self._ctx['end_agent']
        final_ori = tuple(self._ctx['final_ori'])
        correct_obs = str(correct_answer)
        wrong_selected: List[str] = []

        # orientation variants
        wrong_ori: List[str] = []
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if tuple(ori) == tuple(final_ori):
                continue
            a = end_agent.copy(); a.ori = np.array(ori)
            wrong_ori.append(self._observe_text(a, max_items=3))

        # position variants: teleport to other nearby objects, optionally randomize orientation
        wrong_pos: List[str] = []
        final_pos = end_agent.pos.copy()
        others = [o for o in self.room.all_objects if not np.allclose(o.pos, final_pos)]
        others.sort(key=lambda o: float(np.linalg.norm(o.pos - final_pos)))
        for o in others:
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

        # filter with correct_all set and exclude any choice that mentions the origin object
        correct_all = set(self._observe_relationships(end_agent))
        origin_name = str(self._ctx.get('origin_name', ''))

        def valid(s: str) -> bool:
            return bool(s) and (origin_name not in s) and self._is_wrong_forward(s, correct_all)

        for cat in (wrong_ori, wrong_pos, wrong_mut):
            cands = [s for s in cat if valid(s) and (s != correct_obs) and (s not in wrong_selected)]
            if cands:
                pick = str(self.np_random.choice(cands))
                if pick not in wrong_selected:
                    wrong_selected.append(pick)

        # fallback from combined pool
        pool = [s for s in (wrong_ori + wrong_pos + wrong_mut) if valid(s) and (s != correct_obs) and (s not in wrong_selected)]
        self.np_random.shuffle(pool)
        for s in pool:
            if len(wrong_selected) >= 3: break
            wrong_selected.append(s)

        # ultimate fallback: randomize orientation until filled
        while len(wrong_selected) < 3:
            a = end_agent.copy(); a.ori = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)][int(self.np_random.integers(0, 4))])
            s = self._observe_text(a, max_items=3)
            if valid(s) and (s != correct_obs) and (s not in wrong_selected):
                wrong_selected.append(s)

        choices = [correct_obs] + wrong_selected[:3]
        self.np_random.shuffle(choices)
        return choices, int(choices.index(correct_obs))
