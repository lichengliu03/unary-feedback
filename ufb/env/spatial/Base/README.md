# Base Module - Technical Documentation

Core implementation components for the spatial reasoning environment.

## Directory Structure

```
Base/
├── core/                      # Data structures and fundamental classes
│   ├── object.py             # Object and Agent classes
│   ├── room.py               # Room state management
│   ├── relationship.py       # Spatial relationship system
│   ├── graph.py              # Directional graph for tracking relationships
│   └── constant.py           # Constants and configurations
├── actions/                   # Action system
│   ├── base.py               # Abstract base action class
│   └── actions.py            # Concrete action implementations
├── managers/                  # High-level management
│   ├── exploration_manager.py # Exploration phase logic
│   └── evaluation_manager.py  # Evaluation tasks and scoring
├── evaluation/                # Evaluation system
│   ├── tasks.py              # Task implementations
│   └── task_types.py         # Task type definitions and factory
└── utils/                     # Utility functions
    ├── room_utils.py         # Room generation
    └── eval_utilities.py     # Evaluation helpers
```

## Key Implementation Features

### Spatial Relationship System
- 8-directional compass system (N, NE, E, SE, S, SW, W, NW)
- Agent-centric queries ("left", "right", "front", "back")
- Graph-based relationship tracking

### Action System
- Modular action classes inheriting from `BaseAction`
- Structured `ActionResult` return values
- Text-based action sequence parsing

## Usage

### Basic Import
```python
from ragen.env.spatial.Base import Room, ExplorationManager, RoomGenerator
```

### Room Generation
```python
import numpy as np
np_random = np.random.RandomState(42)
room, agent = RoomGenerator.generate_room(n_objects=5, np_random=np_random)
"""Agent is decoupled from Room."""
```

### Exploration
```python
exploration_manager = ExplorationManager(room, agent)
action_seq = ActionSequence.parse("Actions: [JumpTo(table), Observe()]")
result, info = exploration_manager.execute_action_sequence(action_seq)
```

### Evaluation
```python
from ragen.env.spatial.Base.evaluation import create_evaluation_task

task = create_evaluation_task("direction", room, {"difficulty": "medium"})
eval_manager = EvaluationManager(room, [task])
question = eval_manager.get_current_question()
score = eval_manager.submit_answer("north")
```

## Extension Guidelines

### Adding New Actions
1. Create class in `actions/actions.py` inheriting from `BaseAction`
2. Implement `execute(room, agent)` method returning `ActionResult`
3. Update `ActionSequence.parse()` for text parsing
4. Add to exports in `actions/__init__.py`

### Adding New Evaluation Tasks
1. Create task class in `evaluation/tasks.py` inheriting from `BaseEvaluationTask`
2. Implement `generate_question()` method
3. Add new task to `EvalTaskType` enum in `evaluation/task_types.py`
4. Add configuration options as needed



## TODO
1. Change from single room to multiple rooms
2. Modify action:
    a. visibility of objects: can only see objects in same room; a field of view
    b. add GoThroughDoor action: the action will: move agent to the door; move agent into another room
3. Update exploration manager:
    a. gates are also used in the graph

Storyline:
1. Use different exploration strategies as different agents:
- (Oracle) passive agent with minimal queries: only distance queries --> minimal turns (unreachable lower bounds)
- (Strategist) passive agent with best strtagey --> best turns
- (Inquisitor) passive agent with all necessary queries --> maximum turns
- active agent --> #turns for exploration efficiency
2. Exploration efficiency is reformulated to:
- Advantage over Inquisitor
- Gap between Oracle/Strategist
