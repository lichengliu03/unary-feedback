"""E2A: object coordinates and orientations identification task."""

from typing import List, Optional, Tuple
import numpy as np

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object
from ..core.relationship import CardinalBinsAllo
from ..utils.utils import hash

class E2AEvaluationTask(BaseEvaluationTask):
    """Given object names, choose correct coordinates and orientations."""

    QUESTION_TEMPLATE = (
        "Treat your starting position as the origin (0, 0), and facing north.\n"
        "Consider the global map coordinates (x right, y up).\n"
        "Choose the option that correctly lists some objects with their coordinates and orientations.\n\n"
        "Answer format: object: ((x, y), <facing direction>) for each object\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        # choices: each option picks its own random subset of objects
        choices, correct_idx = self.generate_choices()
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(choices_text=choices_text)
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    def _get_orientation_string(self, obj: Object) -> str:
        return {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}[tuple(obj.ori)]

    def generate_choices(self) -> Tuple[List[str], int]:
        # helper: pick random subset
        def pick_subset(k: Optional[int] = None) -> List[Object]:
            pool = list(self.room.objects)
            self.np_random.shuffle(pool)
            k = k or int(self.np_random.integers(3, min(6, len(pool)) + 1)) 
            return pool[:k]

        # correct option
        objs_c = pick_subset(3)
        correct_answer = [
            (o.name, tuple(map(int, o.pos)), self._get_orientation_string(o))
            for o in objs_c
        ]
        correct_str = self._format_answer(correct_answer)
        choices, seen = [correct_str], {correct_str}

        # wrong options: independent subsets, mutate pos/orientation for ≥1 item
        def mutate_positions(coords, idxs, objs):
            origin = tuple(map(int, self.agent.init_pos))
            new_coords = list(coords)
            for i in idxs:
                o = objs[i]
                rid_i = int(getattr(o, 'room_id', 1) or 1)
                p = self._sample_point_with_discrete_change(
                    reference_pos=coords[i],
                    anchor_pos=origin,
                    room_id=rid_i,
                    min_distance=2.0,
                    bin_system=CardinalBinsAllo(),
                    anchor_ori=(0, 1),
                    must_be_free=False,
                ) or coords[i]
                new_coords[i] = p
            return new_coords

        def mutate_orientations(orients, idxs):
            labels = ["north", "east", "south", "west"]
            out = list(orients)
            for i in idxs:
                out[i] = self.np_random.choice([l for l in labels if l != out[i]])
            return out

        while len(choices) < 4:
            objs = pick_subset(3)
            base = [(o.name, tuple(map(int, o.pos)), self._get_orientation_string(o)) for o in objs]
            coords, oris = [c for (_, c, _) in base], [r for (_, _, r) in base]
            mode = int(self.np_random.integers(0, 3))  # 0: pos, 1: ori, 2: both
            if mode in (0, 2):
                idxs = self.np_random.choice(len(base), size=int(self.np_random.integers(1, len(base) + 1)), replace=False)
                coords = mutate_positions(coords, idxs, objs)
            if mode in (1, 2):
                idxs = self.np_random.choice(len(base), size=int(self.np_random.integers(1, len(base) + 1)), replace=False)
                oris = mutate_orientations(oris, idxs)
            cand = [(n, c, r) for (n, *_), c, r in zip(base, coords, oris)]
            s = self._format_answer(cand)
            if s not in seen:
                choices.append(s); seen.add(s)

        self.np_random.shuffle(choices)
        return choices, choices.index(correct_str)

    def _format_answer(self, answer: List[Tuple]) -> str:
        # convert absolute coords to relative to start position
        ox, oy = tuple(map(int, self.agent.init_pos))
        rel_items = []
        for name, abs_coord, orientation in answer:
            rx, ry = int(abs_coord[0]) - ox, int(abs_coord[1]) - oy
            rel_items.append(f"{name}: (({rx}, {ry}), {orientation})")
        return ", ".join(rel_items)
