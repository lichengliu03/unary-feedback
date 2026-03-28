"""Rotation-related evaluation tasks."""

from typing import List, Tuple, Any
import numpy as np
from typing_extensions import override

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object
from ..core.relationship import PairwiseRelationship
from ..utils.utils import hash

class RotEvaluationTask(BaseEvaluationTask):
    """Ask the sequence of objects appearing when rotating in place."""

    QUESTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "You will perform a full 360-degree rotation by continuously turning {turn_direction} in place.\n"
        "Assume all walls are removed (you can see through walls), so every object is visible.\n"
        "Your task is to identify the correct sequence of objects that will appear directly in front of you during the rotation.\n"
        "If two objects have the exact same bearing, list the nearer first.\n\n"
        "Each option below shows a subset of objects in a specific order.\n"
        "Choose the option that shows the correct sequence:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    # ---------- helpers ----------
    def _get_object_info(self, obj: Object, turn_dir: str) -> Tuple[float, float]:
        """Get angle and distance for an object relative to agent position and turn direction."""
        bearing = float(PairwiseRelationship.get_bearing_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)))
        distance = float(PairwiseRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value)
        angle = (bearing % 360.0) if turn_dir == "clockwise" else ((-bearing) % 360.0)
        return angle, distance

    def _sorted_pts(self, turn_dir: str) -> List[Tuple[str, float, float]]:
        pts = []
        for o in self.room.objects:
            if not np.array_equal(o.pos, self.agent.pos):
                ang, dist = self._get_object_info(o, turn_dir)
                pts.append((o.name, ang, dist))
        pts.sort(key=lambda x: (x[1], x[2]))  # by angle, tie -> nearer first
        return pts

    def _greedy_from(self, pts, start_idx: int, eps: float) -> Tuple[List[str], List[float]]:
        n = len(pts)
        names, angs = [], []
        last = None
        for t in range(n):  # one full wrap
            j = (start_idx + t) % n
            name, ang, _ = pts[j]
            if last is None or ((ang - last) % 360.0) > eps:
                names.append(name); angs.append(ang); last = ang
        if (angs[0] - angs[-1]) % 360.0 < eps:
            angs.pop(); names.pop()
        # normalize: start from smallest angle (e.g., [90,180,270,45] -> [45,90,180,270] for CW)
        k = int(np.argmin(angs))
        return names[k:] + names[:k], angs[k:] + angs[:k]

    def _gen_valid_sequence(self, turn_dir: str, eps: float) -> List[str]:
        pts = self._sorted_pts(turn_dir)
        assert len(pts) >= 3, "Need at least 3 objects"
        tries, cur_eps = 0, float(eps)
        while tries < 10:
            start = int(self.np_random.integers(0, len(pts)))
            names, angs = self._greedy_from(pts, start, cur_eps)
            if len(names) >= 3:
                return names[:self.np_random.integers(3, min(len(names), 7) + 1)]
            tries += 1
        # fallback: tighten epsilon and try once more
        cur_eps = min(cur_eps, 1.0)
        print(f"[Rotation Task] Fallback: tighten epsilon to {cur_eps}")
        start = int(self.np_random.integers(0, len(pts)))
        names, angs = self._greedy_from(pts, start, cur_eps)
        assert len(names) >= 3, "Increase object count or decrease angle_eps"
        return names[:self.np_random.integers(3, min(len(names), 7) + 1)]

    def _corrupt_seq(self, seq: List[str]) -> List[str]:
        """Make a wrong sequence from an independently valid one (do NOT touch correct)."""
        rnd, m = self.np_random, len(seq)
        t = int(rnd.integers(0, 4)) # error types
        if t in (0, 3):  # rotate
            k = int(rnd.integers(1, m))
            return seq[k:] + seq[:k]
        if t in (1, 3):  # swap two
            i, j = rnd.choice(m, size=2, replace=False).tolist()
            s = seq[:]; s[i], s[j] = s[j], s[i]; return s
        if t in (2, 3):  # reverse
            return seq[::-1]
        return seq

    def generate_choices(self, correct_answer: List[str]) -> Tuple[List[str], int]:
        fmt = lambda s: ", ".join(s)

        choices, seen = [fmt(correct_answer)], {tuple(correct_answer)}
        while len(choices) < int(self.config.get("num_choices", 4)):
            base = self._gen_valid_sequence(self.turn_direction, self.angle_eps) # valid sequence
            wrong = self._corrupt_seq(base) # then corrupt -> wrong
            if tuple(wrong) not in seen and wrong != correct_answer:
                choices.append(fmt(wrong)); seen.add(tuple(wrong))

        self.np_random.shuffle(choices)
        return choices, choices.index(fmt(correct_answer))

    # ---------- main ----------
    @retry_generate_question
    def generate_question(self) -> str:
        self.turn_direction = self.np_random.choice(["clockwise", "counterclockwise"])
        self.angle_eps = float(self.config.get("angle_eps", 30.0))

        correct_seq = self._gen_valid_sequence(self.turn_direction, self.angle_eps)
        choices, idx = self.generate_choices(correct_seq)
        choices_text, correct_label = self.format_choices(choices, idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            turn_direction=self.turn_direction, choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    @override
    def to_string(self) -> str:
        return f"{self.__class__.__name__}({self.turn_direction})"


class RotDualEvaluationTask(RotEvaluationTask):
    """Given the appearing sequence, ask the rotation direction. TODO: different sequences in each option"""

    QUESTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "You performed a complete 360° rotation in place.\n"
        "During the rotation, these objects appeared directly in front of you in this order:\n"
        "{object_sequence}\n\n"
        "Based on this sequence, in which direction did you rotate?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        self.turn_direction = self.np_random.choice(["clockwise", "counterclockwise"])
        self.angle_eps = float(self.config.get("angle_eps", 30.0))

        correct_seq = self._gen_valid_sequence(self.turn_direction, self.angle_eps)
        object_sequence = ", ".join(correct_seq)

        choices, idx = self.generate_choices(self.turn_direction)
        choices_text, correct_label = self.format_choices(choices, idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_sequence=object_sequence, choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        correct = str(correct_answer)
        other = "counterclockwise" if correct == "clockwise" else "clockwise"
        choices = [correct, other]
        self.np_random.shuffle(choices)
        return choices, choices.index(correct)