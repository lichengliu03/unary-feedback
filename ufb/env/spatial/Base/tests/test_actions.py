import numpy as np
from tos_base.actions.actions import (
    MoveAction, RotateAction, ReturnAction, ObserveAction,
    QueryAction, TermAction, ActionSequence
)
from tos_base.core.object import Object, Agent
from tos_base.core.room import Room
from tos_base.managers.exploration_manager import ExplorationManager


class TestMoveAction:
    """Test MoveAction class"""

    def test_move_action_creation(self):
        """Test MoveAction creation"""
        action = MoveAction("table")
        assert action.target == "table"
        assert str(action) == "JumpTo(table)"

    def test_move_action_with_underscore(self):
        """Test target names with underscores"""
        action = MoveAction("dining_table")
        assert action.target == "dining table"  # Underscores should be replaced with spaces

    def test_move_action_execute_not_found(self):
        """Test moving to non-existent object"""
        action = MoveAction("nonexistent")

        # Create simple test environment
        objs = [Object('table', np.array([1, 2]), np.array([0, 1]))]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)

        result = action.execute(room, agent)
        assert not result.success
        assert "object not found" in result.message


class TestRotateAction:
    """Test RotateAction class"""

    def test_rotate_action_creation(self):
        """Test RotateAction creation"""
        action = RotateAction(90)
        assert action.degrees == 90
        assert str(action) == "Rotate(90)"

    def test_rotate_action_valid_degrees(self):
        """Test valid rotation degrees"""
        valid_degrees = [0, 90, 180, 270, -90, -180, -270]
        for deg in valid_degrees:
            action = RotateAction(deg)
            assert action.degrees == deg

    def test_rotate_action_execute_valid(self):
        """Test valid rotation execution"""
        action = RotateAction(90)

        # Create test environment with at least one object
        objs = [Object('table', np.array([1, 2]), np.array([0, 1]))]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)

        original_ori = agent.ori.copy()
        result = action.execute(room, agent)

        assert result.success
        assert "clockwise 90°" in result.message
        # Verify orientation actually changed
        assert not np.array_equal(agent.ori, original_ori)

    def test_rotate_action_execute_invalid(self):
        """Test invalid rotation degrees"""
        action = RotateAction(45)  # Invalid degree

        objs = [Object('table', np.array([1, 2]), np.array([0, 1]))]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)

        result = action.execute(room, agent)
        assert not result.success
        assert "only" in result.message and "allowed" in result.message


class TestReturnAction:
    """Test ReturnAction class"""

    def test_return_action_creation(self):
        """Test ReturnAction creation"""
        action = ReturnAction()
        assert str(action) == "Return()"

    def test_return_action_execute(self):
        """Test return to initial position"""
        action = ReturnAction()

        objs = [Object('table', np.array([1, 2]), np.array([0, 1]))]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)

        # Move to different position
        agent.pos = np.array([3, 3])
        agent.ori = np.array([1, 0])

        result = action.execute(room, agent)

        assert result.success
        assert "returned to starting position" in result.message
        assert np.array_equal(agent.pos, agent.init_pos)
        assert np.array_equal(agent.ori, agent.init_ori)


class TestObserveAction:
    """Test ObserveAction class"""

    def test_observe_action_creation(self):
        """Test ObserveAction creation"""
        action = ObserveAction()
        assert str(action) == "Observe()"
        assert action.is_final()

    def test_observe_action_execute_empty_view(self):
        """Test observation with no objects in view"""
        action = ObserveAction()

        # Create room with objects completely behind the agent (180 degrees away)
        # Agent faces north [0,1], put object south of agent
        objs = [Object('behind_table', np.array([1, 0]), np.array([0, 1]))]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)

        result = action.execute(room, agent)

        assert result.success
        assert "No objects in field of view" in result.message

    def test_observe_action_execute_with_objects(self):
        """Test observation with objects present"""
        action = ObserveAction()

        # Create test environment with objects
        objs = [
            Object('table', np.array([1, 2]), np.array([0, 1])),
            Object('chair', np.array([2, 2]), np.array([0, 1]))
        ]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)

        result = action.execute(room, agent)

        assert result.success
        assert "table" in result.message
        assert "chair" in result.message


class TestQueryAction:
    """Test QueryAction class"""

    def test_query_action_creation(self):
        """Test QueryAction creation"""
        action = QueryAction("table")
        assert action.obj == "table"
        assert str(action) == "Query(table)"
        assert action.is_final()

    def test_query_action_execute_not_found(self):
        """Test querying non-existent object"""
        action = QueryAction("nonexistent")

        objs = [Object('table', np.array([1, 2]), np.array([0, 1]))]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)

        result = action.execute(room, agent)
        assert not result.success
        assert "object not found" in result.message


class TestTermAction:
    """Test TermAction class"""

    def test_term_action_creation(self):
        """Test TermAction creation"""
        action = TermAction()
        assert str(action) == "Term()"
        assert action.is_final()
        assert action.is_term()

    def test_term_action_execute(self):
        """Test termination action execution"""
        action = TermAction()

        objs = [Object('table', np.array([1, 2]), np.array([0, 1]))]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)

        result = action.execute(room, agent)

        assert result.success
        assert "terminated" in result.message.lower()


class TestActionSequence:
    """Test ActionSequence class"""

    def test_action_sequence_parse_simple(self):
        """Test parsing simple action sequence"""
        seq = ActionSequence.parse("Actions: [Observe()]")
        assert seq is not None
        assert len(seq.motion_actions) == 0
        assert isinstance(seq.final_action, ObserveAction)

    def test_action_sequence_parse_complex(self):
        """Test parsing complex action sequence"""
        seq = ActionSequence.parse("Actions: [JumpTo(table), Rotate(90), Observe()]")
        assert seq is not None
        assert len(seq.motion_actions) == 2
        assert isinstance(seq.motion_actions[0], MoveAction)
        assert isinstance(seq.motion_actions[1], RotateAction)
        assert isinstance(seq.final_action, ObserveAction)

    def test_action_sequence_parse_invalid(self):
        """Test parsing invalid action sequences"""
        # No final action
        seq = ActionSequence.parse("Actions: [JumpTo(table)]")
        assert seq is None

        # Multiple final actions
        seq = ActionSequence.parse("Actions: [Observe(), Query(table)]")
        assert seq is None

    def test_action_sequence_parse_term_only(self):
        """Test sequence with only termination action"""
        seq = ActionSequence.parse("Actions: [Term()]")
        assert seq is not None
        assert len(seq.motion_actions) == 0
        assert isinstance(seq.final_action, TermAction)


class TestIntegration:
    """Integration tests"""

    def test_full_exploration_scenario(self):
        """Test complete exploration scenario"""
        # Create test environment
        objs = [
            Object('table', np.array([1, 2]), np.array([0, 1])),
            Object('chair', np.array([2, 2]), np.array([0, 1])),
            Object('lamp', np.array([1, 3]), np.array([0, 1])),
        ]
        mask = np.ones((6, 6), dtype=np.int8)
        room = Room(objects=objs, mask=mask, name='test_room')
        agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)
        mgr = ExplorationManager(room, agent)

        # Test observation
        seq = ActionSequence.parse("Actions: [Observe()]")
        assert seq is not None
        results = mgr.execute_action_sequence(seq)
        assert len(results) == 1
        assert results[0].success

        # Test move + observation
        seq = ActionSequence.parse("Actions: [JumpTo(table), Observe()]")
        assert seq is not None
        results = mgr.execute_action_sequence(seq)
        assert len(results) == 2

        # Test query
        seq = ActionSequence.parse("Actions: [Query(table)]")
        assert seq is not None
        results = mgr.execute_action_sequence(seq)
        assert len(results) == 1

        # Verify statistics
        assert mgr.action_counts['observe'] > 0
        assert mgr.action_cost > 0


