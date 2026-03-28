"""Simple Evaluation Manager for SpatialGym Environment"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..evaluation.task_types import EvalTaskType
from ..core.room import Room
from ..core.object import Agent
from ..evaluation.tasks import BaseEvaluationTask, EvaluationData

@dataclass
class EvaluationTurnLog:
    """Log data for a single evaluation turn."""
    task_type: str
    user_answer: str
    is_correct: bool
    evaluation_info: Dict[str, Any]
    evaluation_data: EvaluationData
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None

    def to_dict(self):
        evaluation_data = self.evaluation_data.to_dict()
        if "question" in evaluation_data:
            evaluation_data.pop("question")
        evaluation_data['choices'] = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(evaluation_data['choices'])])
        return {
            "task_type": self.task_type,
            "user_answer": self.user_answer,
            "is_correct": self.is_correct,
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "evaluation_info": self.evaluation_info,
            "evaluation_data": evaluation_data
        }


class EvaluationManager:
    """
    Manages evaluation tasks for the SpatialGym environment.
    
    Handles task initialization, question generation, answer evaluation,
    and tracking of evaluation results across multiple tasks.
    """
    DEFAULT_EVAL_SUMMARY = {
        "accuracy": 0.0,
        "total_tasks": 0,
        "correct_count": 0,
        "incorrect_count": 0,
        "unanswered_count": 0
    }
    
    def __init__(self, eval_tasks: List[Dict[str, Any]], np_random: np.random.Generator, room: Room, agent: Agent, history_manager=None, seed: int | None = None):
        # In current implementation, only one evaluation task is allowed
        assert len(eval_tasks) == 1, "Only one evaluation task is supported"
        self.history_manager = history_manager
        self.eval_tasks = []
        spec = eval_tasks[0]
        ttype = spec['task_type']
        num = int(spec.pop('num', 1))
        counts = self.history_manager.get_eval_counts() if self.history_manager else {}
        done_count = 0
        if counts:
            class_name = EvalTaskType.from_short_name(ttype).class_name
            done_count = counts.get(class_name, 0)

        if num > done_count:
            self.eval_tasks = [spec]
        self.np_random = np_random
        self.results = []
        self.turn_logs: List[EvaluationTurnLog] = []
        self.seed = seed
        
        # Initialize tasks
        self.tasks = []
        for idx, task_spec in enumerate(self.eval_tasks):
            task_type = task_spec['task_type']
            task = EvalTaskType.create_task(task_type, np.random.default_rng(int(self.seed)), room.copy(), agent.copy(), {}, history_manager)
            self.tasks.append(task)
            self.results.append({
                "task_type": task.__class__.__name__,
                "correct": False,
                "info": {}
            })
        self.current_index = 0
    
    def _get_current_eval_task(self) -> Optional[BaseEvaluationTask]:
        """Get current evaluation task."""
        assert self.current_index < len(self.tasks), "No more tasks"
        return self.tasks[self.current_index]
    
    def get_current_question(self) -> Optional[str]:
        """Get question for current task."""
        task = self._get_current_eval_task()
        return None if task is None else task.question if task.question else task.generate_question()
    
    def evaluate_answer(self, answer: str) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate answer for current task."""
        assert self.current_index < len(self.tasks), "No more tasks"
        
        task = self.tasks[self.current_index]
        correct, info = task.evaluate(answer)
        
        # Record result
        self.results[self.current_index]["correct"] = correct
        self.results[self.current_index]["info"] = info
        
        # Create turn log
        turn_log = EvaluationTurnLog(
            task_type=task.__class__.__name__,
            user_answer=answer,
            is_correct=correct,
            room_state=task.room,
            agent_state=task.agent,
            evaluation_info=info,
            evaluation_data=task.eval_data
        )
        self.turn_logs.append(turn_log)
        
        return correct, info

    def next_task(self) -> bool:
        """Move to next task. Returns True if there are more tasks."""
        self.current_index += 1
        return self.current_index < len(self.tasks)
    
    def get_last_room_state(self) -> Tuple[Room, Agent]:
        """Get current room and agent state."""
        task = self.tasks[self.current_index - 1]
        return task.room, task.agent
    
    def get_eval_summary(self) -> Dict[str, Any]:
        """Calculate evaluation summary from turn logs."""
        total_tasks = len(self.tasks)
        answered_tasks = len(self.turn_logs)
        unanswered_count = total_tasks - answered_tasks
        correct_count = sum(1 for log in self.turn_logs if log.is_correct)
        incorrect_count = answered_tasks - correct_count
        
        return {
            "accuracy": correct_count / total_tasks if total_tasks > 0 else 0.0,
            "total_tasks": total_tasks,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "unanswered_count": unanswered_count
        }

    def check_and_prune_completed_tasks(self) -> bool:
        """If no tasks to run, return True to signal finished."""
        return len(self.tasks) == 0
    
    # ---------------- Aggregations ----------------
    @staticmethod
    def aggregate_per_sample(env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate evaluation metrics within one sample (counts and accuracy)."""
        tasks = env_data.get('evaluation_tasks') or {}
        per_task = {}
        for task_type, questions in tasks.items():
            n_total = len(questions)
            n_correct = sum(1 for q in questions.values() if q.get('evaluation_log', {}).get('is_correct'))
            per_task[task_type] = {
                'n_total': n_total,
                'n_correct': n_correct,
                'avg_accuracy': (n_correct / n_total) if n_total else 0.0,
            }
        # temp code, TODO
        filtered_per_task = {t: v for t, v in per_task.items() if t not in ['PovEvaluationTask', 'BackwardPovEvaluationTask', 'FalseBeliefDirectionPov', 'DirectionPov']}

        total = sum(v['n_total'] for v in filtered_per_task.values())
        correct = sum(v['n_correct'] for v in filtered_per_task.values())
        return {
            'overall': {'n_total': total, 'n_correct': correct, 'avg_accuracy': (correct / total) if total else 0.0},
            'per_task': per_task,
        }

    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict] = None) -> Dict[str, Any]:
        """Group aggregation; use precomputed per-sample metrics or reuse per-sample aggregation."""
        if not env_data_list:
            return {'avg_accuracy': 0.0, 'task_metrics': {}}

        # temp code, TODO
        for s in env_data_list:
            metrics = s.get('metrics')
            if metrics is None or not isinstance(metrics, dict):
                s['metrics'] = {}
                metrics = s['metrics']
            metrics['evaluation'] = EvaluationManager.aggregate_per_sample(s)

        per_samples = [((s.get('metrics') or {}).get('evaluation') or {}) for s in env_data_list]
        total_count = sum(int(m.get('overall',{}).get('n_total', 0)) for m in per_samples)
        total_correct = sum(int(m.get('overall',{}).get('n_correct', 0)) for m in per_samples)
        agg_task: Dict[str, Dict[str, int]] = {}
        for m in per_samples:
            for t, tm in (m.get('per_task') or {}).items():
                d = agg_task.setdefault(t, {'total': 0, 'correct': 0})
                d['total'] += int(tm.get('n_total', 0))
                d['correct'] += int(tm.get('n_correct', 0))
        task_metrics = {t: {
            'accuracy': (v['correct'] / v['total']) if v['total'] else 0.0,
            'total_count': v['total'],
            'correct_count': v['correct'],
        } for t, v in agg_task.items()}
        return {'avg_accuracy': (total_correct / total_count) if total_count else 0.0, 'task_metrics': task_metrics}
    
    def reset(self):
        """Reset to start."""
        self.current_index = 0
        self.results = []
    



if __name__ == "__main__":
    pass