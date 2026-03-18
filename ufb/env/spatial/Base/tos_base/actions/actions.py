from typing import Optional, List
import re
import numpy as np

from .base import BaseAction, ActionResult
from ..core.object import Gate
from ..core.relationship import PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship, RelationTriple, OrientationRel, DegreeRel

"""
Specific action implementations for spatial exploration.
Contains all concrete action classes and the ActionSequence parser.
"""


from ..utils.utils import ANSWER_LABEL

ACTION_REMINDER = 'To answer the question, you must output "<answer >Actions: [ <M>* <F> ] </answer>" to explore the room to gather info.'

ACTION_INSTRUCTION = """\
Use "Actions: [ <M>* <F> ]" to explore the room.
<M>: JumpTo(OBJ) | Rotate(DEG)
<F>: Observe() | Term(ANSWER)

Rules:
- Start with zero or more <M>.
- End with exactly one <F>.
- At most one Observe().
- Term(ANSWER) must be alone.

Available Actions:
{actions}

Examples:
{examples}
"""



class MoveAction(BaseAction):
    """Jump to a target object"""
    
    format_desc = "JumpTo(OBJ)"
    description = "Jump to a visible object in Field of View. No orientation change."
    example = "JumpTo(table)"
    format_pattern = r"^JumpTo\(([A-Za-z0-9_ -]+)\)$"
    cost = 0
    
    def __init__(self, target: str):
        super().__init__(target)
        self.target = target.replace('_', ' ')
    
    def success_message(self, **kwargs) -> str:
        extra = kwargs.get('extra')
        return f"You jumped to {self.target}." + (f" {extra}" if extra else "")
    
    def error_message(self, error_type: str) -> str:
        errors = {"not_found": "object not found", "not_visible": "object not visible", "not_observed": "object not observed yet"}
        return f"Cannot jump to '{self.target}': {errors.get(error_type, 'execution failed')}."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute jump action on room state."""
        if not room.has_object(self.target):
            return ActionResult(False, self.get_feedback(False, "not_found"), str(self), 'move', {'target_name': self.target})
        
        target_obj = room.get_object_by_name(self.target)
        observed_items = set(kwargs.get('observed_items', [])) # if None, all objects are observed
        if observed_items is not None and self.target not in observed_items:
            return ActionResult(False, self.get_feedback(False, "not_observed"), str(self), 'move', {'target_name': self.target})
        if not kwargs.get('move_anyway', False) and not self._is_visible(agent, target_obj):
            return ActionResult(False, self.get_feedback(False, "not_visible"), str(self), 'move', {'target_name': self.target})        
        
        # extra messaging for gates and entering rooms
        prev_room = agent.room_id
        extra_msg = f"This gate connects rooms {target_obj.room_id[0]} and {target_obj.room_id[1]}." if isinstance(target_obj, Gate) else None
        
        # apply move and room membership
        agent.pos, agent.room_id = target_obj.pos, target_obj.room_id
        
        # entering room from a gate
        if (extra_msg is None) and (not isinstance(target_obj, Gate)) and isinstance(prev_room, (list, tuple)):
            extra_msg = f"You are in room {target_obj.room_id}."
        return ActionResult(True, self.get_feedback(True, extra=extra_msg), str(self), 'move', {'target_name': self.target})
    
    def __repr__(self):
        return f"JumpTo({self.target})"

class RotateAction(BaseAction):
    """Rotate by specified degrees"""
    
    format_desc = "Rotate(DEG)"
    description = "Rotate relative to current orientation. Valid: -270, -180, -90, 0, 90, 180, 270."
    example = "Rotate(-90)"
    format_pattern = r"^Rotate\(([0-9-]+)\)$"
    VALID_DEGREES = [0, 90, 180, 270, -90, -180, -270]
    
    def __init__(self, degrees: int):
        super().__init__(degrees)
        self.degrees = int(degrees)
        
    def success_message(self, **kwargs) -> str:
        if self.degrees == 0:
            return "You rotated 0°."
        direction = 'clockwise' if self.degrees > 0 else 'counterclockwise'
        return f"You rotated {direction} {abs(self.degrees)}°."
    
    def error_message(self, error_type: str) -> str:
        if error_type == "invalid_degree":
            return f"Cannot rotate by {self.degrees}°: only {self.VALID_DEGREES} allowed."
        return f"Cannot rotate by {self.degrees}°: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute rotate action on room state."""
        if self.degrees is None or self.degrees not in self.VALID_DEGREES:
            return ActionResult(False, self.get_feedback(False, "invalid_degree"), str(self), 'rotate', {'degrees': self.degrees})
        agent.ori = agent.ori @ self._get_rotation_matrix(self.degrees)  
        return ActionResult(True, self.get_feedback(True), str(self), 'rotate', {'degrees': self.degrees})
    
    def __repr__(self):
        return f"Rotate({self.degrees})"


class ReturnAction(BaseAction):
    """Return to anchor position"""
    
    format_desc = "Return()"
    description = "Return to start pose."
    example = "Return()"
    format_pattern = r"^Return\(\)$"
    cost = 0
    
    def success_message(self, **kwargs) -> str:
        return "You returned to starting position and orientation."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot return to anchor: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute return action on room state."""
        # move to initial position
        agent.pos = agent.init_pos.copy()
        # rotate to initial orientation (compute delta)
        # report clockwise degrees from north; ensure consistency with _ori_to_deg in agent_proxy
        deg = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}[tuple(agent.ori)]
        agent.ori = agent.init_ori.copy()
        # restore room id if tracked
        if agent.init_room_id is not None:
            agent.room_id = agent.init_room_id
        return ActionResult(True, self.get_feedback(True), str(self), 'return', {'target_name': 'initial_pos', 'degrees': deg})
    
    def __repr__(self):
        return "Return()"


class ObserveBase(BaseAction):
    """Base observe implementation (internal)."""
    
    format_desc = "Observe()"
    description = "Observe objects in FOV."
    example = "Observe()"
    format_pattern = r"^Observe\(\)$"
    cost = 1
    directional_template = "{obj_name}: {dir_str}"
    orientation_template = "{obj_name} facing {orientation}"
    
    def __init__(self):
        super().__init__()
    
    
    def _collect_obj_observations(self, agent, visible_objects, anchor_name: str, discrete: bool = False):
        relationships: List[str] = []
        relation_triples: List[RelationTriple] = []
        for obj in visible_objects:
            if discrete:
                rel = PairwiseRelationshipDiscrete.relationship(tuple(obj.pos), tuple(agent.pos), anchor_ori=tuple(agent.ori))
            else:
                rel = PairwiseRelationship.relationship(tuple(obj.pos), tuple(agent.pos), anchor_ori=tuple(agent.ori), full=True)
            pairwise_str = rel.to_string()

            if hasattr(obj, 'has_orientation') and not obj.has_orientation:
                answer_str = f"{obj.name}: {pairwise_str}"
            else:
                if isinstance(obj, Gate):
                    rid = agent.room_id
                    if isinstance(rid, (list, tuple)):
                        rid = list(set(agent.room_id) & set(obj.room_id))
                        assert len(rid) == 1, f"intersection of room ids is not unique: {rid}"
                        rid = rid[0]
                    gate_ori = obj.get_ori_for_room(int(rid)) if rid is not None else obj.ori
                    ori_pair = OrientationRel.get_relative_orientation(tuple(gate_ori), tuple(agent.ori))
                    ori_str = OrientationRel.to_string(ori_pair, 'ego', 'orientation', if_gate=True)
                else:
                    ori_pair = OrientationRel.get_relative_orientation(tuple(obj.ori), tuple(agent.ori))
                    ori_str = OrientationRel.to_string(ori_pair, 'ego', 'orientation')
                answer_str = f"{obj.name}: {pairwise_str}, {ori_str}"
            relationships.append(answer_str)
            relation_triples.append(RelationTriple(subject=obj.name, anchor=anchor_name, relation=rel, orientation=tuple(agent.ori)))
        final_answer = "\n" + "\n".join(f"• {rel}" for rel in relationships)
        return final_answer, relationships, relation_triples
    
    def success_message(self, **kwargs) -> str:
        return f"You observe: {kwargs.get('answer', 'nothing')}."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot observe: execution failed."

    

class ObserveAction(ObserveBase):
    """Observe with approximate relations and local (near) pair descriptions"""
    format_desc = "Observe()"
    description = "Report objects and relations in FOV. Must be the last action."
    example = "Observe()"
    format_pattern = r"^Observe\(\)$"
    cost = 1
    @staticmethod
    def is_final() -> bool:
        return True

    def __repr__(self):
        return "Observe()"


    
    def _collect_local_relationships(self, agent, visible_objects, anchor_name: str):
        # proximity-based pair relations using discrete relationship binning
        relationships, relation_triples = [], []
        n = len(visible_objects)
        for i in range(n):
            for j in range(i + 1, n):
                a_obj, b_obj = visible_objects[i], visible_objects[j]
                # NOTE always use agent's orientation for orientation
                prox_rel = ProximityRelationship.from_positions(tuple(a_obj.pos), tuple(b_obj.pos), tuple(agent.ori))
                if prox_rel is not None:
                    relationships.append(prox_rel.to_string(a_obj.name, b_obj.name))
                    relation_triples.append(RelationTriple(subject=a_obj.name, anchor=b_obj.name, relation=prox_rel, orientation=tuple(agent.ori)))
        final_answer = "\n".join(f"• {rel}" for rel in relationships)
        return final_answer, relationships, relation_triples

    def execute(self, room, agent, **kwargs) -> ActionResult:
        neglect_objects = kwargs.get('neglect_objects', []) + [obj.name for obj in room.all_objects if np.allclose(obj.pos, agent.pos)]
        visible_objects = [obj for obj in room.all_objects if self._is_visible(agent, obj) and obj.name not in neglect_objects]
        if not visible_objects:
            answer = "No objects in field of view."
            return ActionResult(True, self.get_feedback(True, answer=answer), str(self), 'observe', {
                'answer': answer, 'visible_objects': [], 'relationships': [], 'local_relationships': []
            })

        anchor_name = self.get_anchor_name(room, agent) if not kwargs.get('free_position', False) else 'free_position'
        pairwise_answer, relationships, pairwise_relation_triples = self._collect_obj_observations(agent=agent, visible_objects=visible_objects, anchor_name=anchor_name, discrete=True)
        local_answer, local_relationships, local_relation_triples = self._collect_local_relationships(agent, visible_objects, anchor_name)

        final_answer = pairwise_answer
        if local_answer:
            final_answer += (
                f"\nAssume your current facing direction is called \"north\", mutually close object relations in your FOV:\n{local_answer}"
            )
        return ActionResult(True, self.get_feedback(True, answer=final_answer), str(self), 'observe', {
            'answer': final_answer,
            'visible_objects': [obj.name for obj in visible_objects],
            'relationships': relationships,
            'local_relationships': local_relationships,
            'relation_triples': pairwise_relation_triples + local_relation_triples
        })

class TermAction(BaseAction):
    """Terminate exploration with an answer"""
    
    format_desc = "Term(ANSWER)"
    description = "Terminate with answer A, B, C, or D."
    example = "Term(A)"
    format_pattern = r"^Term\(([A-D])\)$"
    cost = 0
    
    def __init__(self, answer: str):
        super().__init__(answer)
        self.answer = answer

    def success_message(self, **kwargs) -> str:
        return f"Exploration terminated with answer {self.answer}."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot terminate exploration: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute term action on room state."""
        return ActionResult(True, self.get_feedback(True), str(self), 'term', {'terminated': True, 'answer': self.answer})
    
    @staticmethod
    def is_final() -> bool:
        return True
    
    @staticmethod
    def is_term() -> bool:
        return True
    
    def __repr__(self):
        return f"Term({self.answer})"


# Internal-only forced terminate (not exposed to parser/registry)
class ForcedTermAction(TermAction):
    format_desc = "ForcedTerm()"
    description = "Forced termination when exploration steps are exhausted."
    example = ""
    format_pattern = r"^ForcedTerm\(\)$"
    cost = 0
    def __init__(self):
        super().__init__("A") # Default dummy answer
    def success_message(self, **kwargs) -> str:
        return "Exploration ended. No further exploration actions are allowed."
    def execute(self, room, agent, **kwargs) -> ActionResult:
        return ActionResult(True, self.get_feedback(True), str(self), 'forced_term', {'terminated': True, 'internal': True})
    def __repr__(self): return "ForcedTerm()"


class QueryBase(BaseAction):
    """Base class for query actions."""
    format_desc = "Query(obj)"
    example = "Query(table)"
    format_pattern = r"^Query\(([A-Za-z0-9_ -]+)\)$"
    cost = 2
    def __init__(self, obj: str):
        super().__init__(obj)
        self.obj = obj if obj == 'initial_pos' else obj.replace('_', ' ')
    def error_message(self, error_type: str) -> str:
        return f"Cannot query: {error_type}"
    @staticmethod
    def is_final() -> bool: return True

class QueryAction(QueryBase):
    """Query object coordinates in the initial frame and emit relation triple to initial_pos."""
    description = (
        "Return object's coordinates with agent's initial position as origin, north as y+ axis. "
        "High cost, only use when necessary to eliminate ambiguities."
    )
    def success_message(self, **kwargs) -> str:
        return f"You query {self.obj}: {kwargs.get('answer','unknown')}"
    def __repr__(self): return f"Query({self.obj})"
    def execute(self, room, agent, **kwargs) -> ActionResult:
        if self.obj != 'initial_pos' and (not room.has_object(self.obj)):
            return ActionResult(False, self.get_feedback(False, "object not found"), str(self), 'query', {})
        obj_pos = room.get_object_by_name(self.obj).pos if self.obj != 'initial_pos' else agent.init_pos
        obj_ori = room.get_object_by_name(self.obj).ori if self.obj != 'initial_pos' else agent.init_ori
        v = (obj_pos - agent.init_pos)
        assert np.allclose(agent.init_ori, np.array([0, 1])), "Initial orientation must be north"
        ans_pos = f"({int(v[0])}, {int(v[1])})"
        ori_pair = OrientationRel.get_relative_orientation(tuple(obj_ori), tuple(agent.init_ori))
        ans_ori = OrientationRel.to_string(ori_pair, 'allo', 'orientation')
        ans = ans_pos + ", " + ans_ori
        rel = PairwiseRelationship.relationship(tuple(obj_pos), tuple(agent.init_pos), anchor_ori=tuple(agent.init_ori), full=True)
        return ActionResult(True, self.get_feedback(True, answer=ans), str(self), 'query', {
            'answer': ans,
            'object': self.obj,
            'coords': (int(v[0]), int(v[1])),
            'orientation': ans_ori,
            'relation_triples': [RelationTriple(subject=self.obj, anchor='initial_pos', relation=rel, orientation=tuple(agent.init_ori))]
        })

class QueryRelAction(QueryBase):
    """Legacy: query accurate relationship from current agent pose."""
    format_desc = "QueryRel(obj)"
    description = "Return accurate spatial relationship from current agent pose."
    example = "QueryRel(table)"
    format_pattern = r"^QueryRel\(([A-Za-z0-9_ -]+)\)$"
    def success_message(self, **kwargs) -> str:
        return f"You query {self.obj}: {kwargs.get('answer','unknown')}"
    def __repr__(self): return f"QueryRel({self.obj})"
    def execute(self, room, agent, **kwargs) -> ActionResult:
        if self.obj != 'initial_pos' and (not room.has_object(self.obj)):
            return ActionResult(False, self.get_feedback(False, "object not found"), str(self), 'query', {})
        obj_pos = room.get_object_by_name(self.obj).pos if self.obj != 'initial_pos' else agent.init_pos
        rel = PairwiseRelationship.relationship(tuple(obj_pos), tuple(agent.pos), anchor_ori=tuple(agent.ori), full=True)
        ans = rel.to_string()
        return ActionResult(True, self.get_feedback(True, answer=ans), str(self), 'query', {
            'answer': ans,
            'object': self.obj,
            'relation_triples': [RelationTriple(subject=self.obj, anchor=self.get_anchor_name(room, agent), relation=rel, orientation=tuple(agent.ori))]
        })






# Action registry for easy lookup
# Expose all observe variants; default flows may still prefer ObserveApprox
ACTION_CLASSES = [
    MoveAction, RotateAction, ObserveAction, TermAction
]


class ActionSequence:
    """Sequence of actions for spatial exploration"""
    
    def __init__(self, motion_actions: List[BaseAction] = None, final_action: BaseAction = None):
        self.motion_actions = motion_actions or []
        self.final_action = final_action
    
    def __repr__(self):
        motions = ", ".join(str(action) for action in self.motion_actions)
        return f"ActionSequence(motions=[{motions}], final={self.final_action})"

    @classmethod
    def parse(cls, action_str: str) -> Optional['ActionSequence']:
        m = re.search(r'\[(.*)\]', action_str.strip())
        if not m:
            return None
        # extract top-level actions like JumpTo(table), Rotate(90), Term()
        action_strs = re.findall(r'([A-Za-z]+\([^()]*\))', m.group(1))
        if not action_strs:
            return None

        # Parse all actions
        parsed_actions = []
        for act_s in action_strs:
            act = cls._parse_single_action(act_s.strip())
            if not act:
                return None
            parsed_actions.append(act)

        # Otherwise, standard: zero or more motions then exactly one final
        motions, final_action = [], None
        for i, act in enumerate(parsed_actions):
            if i == len(parsed_actions) - 1:
                if not act.is_final():
                    return None
                final_action = act
            else:
                if act.is_final():
                    return None
                motions.append(act)
        if isinstance(final_action, TermAction):
            if any(not isinstance(a, ReturnAction) for a in motions):
                return None
        return cls(motions, final_action)
    
    @staticmethod
    def _parse_single_action(action_str: str) -> Optional[BaseAction]:
        """Parse a single action string using registered action classes"""
        for action_class in ACTION_CLASSES:
            if action := action_class.parse(action_str):
                return action
        return None
    
    @staticmethod
    def get_usage_instructions(vision: bool = False) -> str:
        """Get usage instructions for action sequences"""
        def _desc(cls):
            if vision and cls is ObserveAction:
                return "Return RGB image of FOV. Must be last."
            return cls.description

        motion_actions = [cls for cls in ACTION_CLASSES if not cls.is_final()]
        final_actions = [cls for cls in ACTION_CLASSES if cls.is_final()]

        action_desc = (
            "Movement:\n" +
            "\n".join(f"- {cls.format_desc}: {_desc(cls)}" for cls in motion_actions) +
            "\nFinal:\n" +
            "\n".join(f"- {cls.format_desc}: {_desc(cls)}" for cls in final_actions)
        )
        examples = (
            "Valid: [JumpTo(table), Rotate(90), Observe()]\n"
            "Valid: [Term(A)]"
        )
        
        return ACTION_INSTRUCTION.format(
            actions=action_desc,
            examples=examples,
            field_of_view=BaseAction.get_field_of_view(),
            answer_label=ANSWER_LABEL
        )
