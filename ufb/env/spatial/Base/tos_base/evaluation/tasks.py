"""
Base evaluation definitions (data and abstract base classes).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from functools import wraps

from ..core.room import Room
from ..core.object import Agent, Object
from ..utils.eval_utilities import multi_choice_eval_fn
from ..actions import RotateAction, ObserveAction
from ..utils.action_utils import action_results_to_text
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    StandardDistanceBins,
)
from ..utils.utils import hash

@dataclass
class EvaluationData:
    id: str
    question: str
    answer: str
    task_type: str
    action: str = None
    choices: List[str] = None
    kwargs: Dict = None

    def __post_init__(self):
        # Lazy import to avoid circular dependency during module import
        from .task_types import EvalTaskType  # type: ignore
        valid_task_types = EvalTaskType.get_class_names()
        assert self.task_type in valid_task_types, f"Invalid task type: {self.task_type}"
        if self.choices is None:
            self.choices = []
    
    def evaluate(self, pred: Any) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate an answer to the given question using multi-choice evaluation"""
        return multi_choice_eval_fn(pred, self.answer), {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation data to a dictionary"""
        return {
            'id': self.id,
            'question': self.question,
            'action': self.action,
            'answer': self.answer,
            'task_type': self.task_type,
            'choices': self.choices,
            'kwargs': self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationData':
        """Initialize the evaluation data from a dictionary"""
        return cls(**data)


class BaseEvaluationTask(ABC):
    """Abstract base class for all spatial evaluation tasks."""
    
    def __init__(self, np_random: np.random.Generator, room: Room, agent: Agent, config: Dict[str, Any] = None, history_manager=None):
        """Initialize the evaluation task"""
        self.config = config or {}
        self.np_random = np_random
        self.room = room.copy()
        self.agent = agent.copy()
        self.history_manager = history_manager
        self.eval_data = EvaluationData(
            id="",
            question="",
            answer="",
            action=None,
            task_type=self.__class__.__name__,
            choices=[],
            kwargs={},
        )

    @property
    def answer(self) -> Any:
        return self.eval_data.answer
    
    @property
    def question(self) -> str:
        return self.eval_data.question
    
    @property
    def choices(self) -> List[str]:
        return self.eval_data.choices
    
    def format_choices(self, choices: List[str], correct_index: int) -> Tuple[str, str]:
        """Format choices as lines and return (choices_text, correct_label)."""
        return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]), chr(65 + correct_index)
    
    @abstractmethod
    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        """Return (choices, correct_index)."""
        raise NotImplementedError
    
    @abstractmethod
    def generate_question(self) -> str:
        """Generate evaluation question based on the current room/agent state."""
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        raise NotImplementedError
    
    def evaluate(self, pred: Any) -> Tuple[bool, Dict[str, Any]]:
        return self.eval_data.evaluate(pred)
    
    def to_string(self) -> str:
        return f"{self.__class__.__name__}()"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation task to a dictionary"""
        return {
            'question': self.question,
            'answer': self.answer,
            'choices': self.choices,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        raise NotImplementedError

    @classmethod
    def create_task_from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        from .task_types import EvalTaskType  # Lazy import
        task_types = EvalTaskType.get_class_map()
        task_type = data.get('type', cls.__name__)
        return task_types.get(task_type, cls).from_dict(data)



    # ---- Shared helper: find a point that changes discrete pairwise relationships ----
    def _sample_point_with_discrete_change(
        self,
        reference_pos: Tuple[int, int],
        anchor_pos: Tuple[int, int],
        room_id: int,
        min_distance: float = 2.0,
        bin_system=None,
        distance_bin_system=None,
        anchor_ori: Tuple[int, int] = (0, 1),
        must_be_free: bool = True,
        max_trials: int = 500,
    ) -> Optional[Tuple[int, int]]:
        """Pick a new point (same room) so the discrete relation to anchor_pos changes.

        Relation is PairwiseRelationshipDiscrete(reference -> anchor). Distance to reference must be ≥ min_distance.
        TODO debug
        """
        if distance_bin_system is None:
            distance_bin_system = StandardDistanceBins()

        def rel_pair(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
            r = PairwiseRelationshipDiscrete.relationship(a, b, anchor_ori=anchor_ori, bin_system=bin_system, distance_bin_system=distance_bin_system)
            return int(r.direction.bin_id), int(r.dist.bin_id)

        base = rel_pair(tuple(map(int, reference_pos)), tuple(map(int, anchor_pos)))
        xmin, xmax, ymin, ymax = self.room.get_boundary(room_id=room_id)
        candidates = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)]
        self.np_random.shuffle(candidates)
        tried = 0
        for x, y in candidates:
            if tried >= max_trials:
                break
            tried += 1
            if (x, y) == tuple(reference_pos):
                continue
            if must_be_free and self.room.get_cell_info(x, y)['object_name']:
                continue
            if float(np.linalg.norm(np.array((x, y)) - np.array(reference_pos))) < float(min_distance) - 1e-6:
                continue
            if rel_pair((int(x), int(y)), tuple(map(int, anchor_pos))) != base:
                return (int(x), int(y))
        # fallback: any free point ≥ min_distance
        for x, y in candidates:
            if must_be_free and self.room.get_cell_info(x, y)['object_name']:
                continue
            if float(np.linalg.norm(np.array((x, y)) - np.array(reference_pos))) >= float(min_distance) - 1e-6:
                return (int(x), int(y))
        return None
    
    def _take_observations(self, neglect_objects: List[str] = None) -> str:
        obs_result = ObserveAction().execute(self.room, self.agent, neglect_objects=neglect_objects or [], free_position=True)
        return action_results_to_text([obs_result])

    def _take_full_observations(self, neglect_objects: List[str] = None) -> str:
        action_results = []
        # Always take 4 views (90° FOV) covering a full 360° turn
        room = self.room.copy()
        agent = self.agent.copy()
        action_results.append(ObserveAction().execute(room, agent, neglect_objects=neglect_objects or [], free_position=True))
        for _ in range(3):
            action_results.append(RotateAction(90).execute(room, agent))
            action_results.append(ObserveAction().execute(room, agent, neglect_objects=neglect_objects or [], free_position=True))
        return action_results_to_text(action_results)



# ---- Decorator: retry question generation with max retries and history de-dup ----
def retry_generate_question(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retry = int(self.config.get('max_retry', 10))
        for _ in range(max_retry):
            q = func(self, *args, **kwargs)
            # Ensure question field is populated
            if isinstance(q, str) and q:
                self.eval_data.question = q
            q = self.eval_data.question
            # Ensure ID exists for history checks
            if not getattr(self.eval_data, 'id', None) or not self.eval_data.id:
                self.eval_data.id = hash(q)
            # Accept if no history manager or not seen
            hm = getattr(self, 'history_manager', None)
            if not hm or not hm.has_question(self.eval_data.id):
                return q
        raise Exception('Failed to generate question')
    return wrapper