"""
Base module for spatial reasoning environment.

This module provides the core components for spatial reasoning tasks including:
- Core data structures (objects, rooms, relationships)
- Action system for agent interactions
- Exploration and evaluation management
- Utility functions for room generation and evaluation
"""

# Core data structures
from .core.object import Object, Agent
from .core.room import Room, BaseRoom
from .core.relationship import DirPair, Dir, PairwiseRelationship
from .core.constant import (
    AGENT_NAME, 
    CANDIDATE_OBJECTS, 
    ADDITIONAL_CANDIDATE_OBJECTS,
    easy_room_config,
    easy_room_config_2,
    easy_room_config_3
)

# Action system
from .actions.base import BaseAction, ActionResult
from .actions.actions import (
    MoveAction,
    RotateAction, 
    ReturnAction,
    ObserveAction,
    TermAction,
    ActionSequence
)

# Managers
from .managers.exploration_manager import ExplorationManager, ExplorationTurnLog
from .managers.agent_proxy import AgentProxy
from .managers.evaluation_manager import EvaluationManager, EvaluationTurnLog
from .managers.cognitive_map_manager import CognitiveMapManager, CognitiveMapTurnLog
from .managers.history_manager import HistoryManager

# Evaluation tasks
from .evaluation.tasks import BaseEvaluationTask
from .evaluation.task_types import EvalTaskType

# Utilities
from .utils.room_utils import RoomGenerator, get_topdown_info, get_room_description, RoomPlotter
from .utils.eval_utilities import *

__all__ = [
    # Core
    'Object', 'Agent', 'Room', 'BaseRoom', 'DirPair', 'PairwiseRelationship', 'Dir',
    'AGENT_NAME', 'CANDIDATE_OBJECTS', 'ADDITIONAL_CANDIDATE_OBJECTS',
    'easy_room_config', 'easy_room_config_2', 'easy_room_config_3',
    
    # Actions
    'BaseAction', 'ActionResult', 'ActionSequence',
    'MoveAction', 'RotateAction', 'ReturnAction', 'ObserveAction', 'TermAction',
    
    # Managers
    'ExplorationManager', 'ExplorationTurnLog', 'EvaluationManager', 'EvaluationTurnLog', 'CognitiveMapManager', 'CognitiveMapTurnLog', 'AgentProxy', 'HistoryManager',
    
    # Evaluation
    'BaseEvaluationTask', 'EvalTaskType',
    
    # Utils
    'RoomGenerator', 'get_topdown_info', 'get_room_description', 'RoomPlotter',
]
