from enum import Enum
from typing import Dict, Type, TYPE_CHECKING
import numpy as np

from ..core.room import Room
from ..core.object import Agent
if TYPE_CHECKING:
    from .tasks import BaseEvaluationTask

class EvalTaskType(Enum):
    """Enum for all available evaluation task types."""
    
    # Task type definitions: (short_name, class_name)
    DIR = ("dir", "DirectionEvaluationTask")
    ROT = ("rot", "RotEvaluationTask")
    ROT_DUAL = ("rot_dual", "RotDualEvaluationTask")
    POV = ("pov", "PovEvaluationTask")
    BWD_POV = ("bwd_pov", "BackwardPovEvaluationTask")
    E2A = ("e2a", "E2AEvaluationTask")
    FWD_LOC = ("fwd_loc", "ForwardLocEvaluationTask")
    BWD_LOC = ("bwd_loc", "BackwardLocEvaluationTask")
    FWD_FOV = ("fwd_fov", "ForwardFOVEvaluationTask")
    BWD_NAV = ("bwd_nav", "BackwardNavEvaluationTask")
    FALSE_BELIEF = ("false_belief", "FalseBeliefDirectionPov")
    DIR_ANCHOR = ("dir_anchor", "DirectionPov")
    
    def __init__(self, short_name: str, class_name: str):
        self.short_name = short_name
        self.class_name = class_name
    
    @classmethod
    def get_short_names(cls) -> list[str]:
        """Get all short names for task types."""
        return [task.short_name for task in cls]
    
    @classmethod
    def get_class_names(cls) -> list[str]:
        """Get all class names for task types."""
        return [task.class_name for task in cls]
    
    @classmethod
    def get_task_map(cls) -> Dict[str, 'Type[BaseEvaluationTask]']:
        """Get mapping from short names to task classes."""
        # Import here to avoid circular imports
        from .direction import DirectionEvaluationTask, PovEvaluationTask, BackwardPovEvaluationTask, DirectionPov
        from .rotation import RotEvaluationTask, RotDualEvaluationTask
        from .e2a import E2AEvaluationTask
        from .localization import ForwardLocEvaluationTask, BackwardLocEvaluationTask
        from .false_belief import FalseBeliefDirectionPov
        from .navigation_tasks import ForwardFOVEvaluationTask, BackwardNavEvaluationTask
        
        task_map = {
            cls.DIR.short_name: DirectionEvaluationTask,
            cls.ROT.short_name: RotEvaluationTask,
            cls.ROT_DUAL.short_name: RotDualEvaluationTask,
            cls.POV.short_name: PovEvaluationTask,
            cls.DIR_ANCHOR.short_name: DirectionPov,
            cls.E2A.short_name: E2AEvaluationTask,
            cls.FWD_LOC.short_name: ForwardLocEvaluationTask,
            cls.BWD_LOC.short_name: BackwardLocEvaluationTask,
            cls.FALSE_BELIEF.short_name: FalseBeliefDirectionPov,
            cls.FWD_FOV.short_name: ForwardFOVEvaluationTask,
            cls.BWD_NAV.short_name: BackwardNavEvaluationTask,
            cls.BWD_POV.short_name: BackwardPovEvaluationTask,
        }
        return task_map
    
    @classmethod
    def get_class_map(cls) -> Dict[str, 'Type[BaseEvaluationTask]']:
        """Get mapping from class names to task classes."""
        task_map = cls.get_task_map()
        return {task.class_name: task_class for task, task_class in 
                zip(cls, task_map.values())}
    
    @classmethod
    def from_short_name(cls, short_name: str) -> 'EvalTaskType':
        """Get task type from short name."""
        for task in cls:
            if task.short_name == short_name:
                return task
        raise ValueError(f"Unknown task short name: {short_name}")
    
    @classmethod
    def from_class_name(cls, class_name: str) -> 'EvalTaskType':
        """Get task type from class name."""
        for task in cls:
            if task.class_name == class_name:
                return task
        raise ValueError(f"Unknown task class name: {class_name}")
    
    @classmethod
    def create_task(cls, task_name: str, np_random: np.random.Generator, room: 'Room', agent: 'Agent', config: dict = None, history_manager = None) -> 'BaseEvaluationTask':
        """Create an evaluation task instance from task name."""
        task_map = cls.get_task_map()
        if task_name in task_map:
            task_class = task_map[task_name]
            return task_class(np_random, room, agent, config or {}, history_manager)
        else:
            raise ValueError(f"Unknown evaluation task: {task_name}") 
        

if __name__ == "__main__":
    from ..utils.room_utils import RoomPlotter, RoomGenerator
    from tqdm import tqdm


    task_name = 'dir_anchor'
    for seed in tqdm(range(0, 1)):
        np_random = np.random.default_rng(seed)
        room, agent = RoomGenerator.generate_room(
            room_size=(15, 15),
            n_objects=10,
            np_random=np_random,
            room_name='room',
            level=0,
            main=12,
        )
        # print(f'room: {room}')
        # print(f'agent: {agent}')
        # RoomPlotter.plot(room, agent, mode='img', save_path='room.png')
        task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
        print(task.generate_question(), task.answer)
        # task.generate_question()
        print(task.answer)