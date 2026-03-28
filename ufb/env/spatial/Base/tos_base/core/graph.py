from typing import Dict, List, Tuple, Optional, Literal
import numpy as np
import copy
from dataclasses import dataclass

from .object import Object
from .relationship import (
    DirPair,
    Dir,
    PairwiseRelationship,
)

@dataclass
class Matrices:
    """Data structure to hold adjacency matrices and their working copies."""
    vertical: np.ndarray
    horizontal: np.ndarray
    vertical_working: np.ndarray
    horizontal_working: np.ndarray
    asked: np.ndarray


class DirectionalGraph:
    """
    Represents spatial relationships between objects in a room using graph.
    
    The class maintains two matrices for tracking vertical and horizontal relationships
    between objects, along with working copies used during updates.

    Two kinds of matrices are used:
    - Final matrices: 
        - (A, B) = 1 means positive direction (front/right)
        - (A, B) = -1 means negative direction (back/left)
        - (A, B) = 0 means same position
    - Working matrices:
        - (A, B) = 1 means there is a directed path from A -> B
        - (A, B) = 0 means no directed path from A -> B
    """
    
    VALID_ROTATION_DEGREES: Tuple[int, ...] = (0, 90, 180, 270)

    def __init__(self, objects: List[Object], is_explore: bool = False) -> None:
        """
        Initialize adjacency matrices for a set of objects.

        Args:
            objects: List of objects to track relationships between
            is_explore: If True, initialize empty matrices, otherwise, all known
        """
        # node directory
        self.nodes: List[str] = [obj.name for obj in objects]
        self.name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.nodes)}
        self.size = len(self.nodes)

        # matrices (final int8, working bool, known bool)
        if not is_explore:
            self._init_from_objects(objects)
        else:
            self._create_empty_state(self.size)

        self.is_explore = is_explore

    def _create_empty_state(self, size: int) -> None:
        """Create empty matrices with int8 values and known masks."""
        self._v_matrix = np.zeros((size, size), dtype=np.int8)
        self._h_matrix = np.zeros((size, size), dtype=np.int8)
        self._v_matrix_working = np.zeros((size, size), dtype=bool)
        self._h_matrix_working = np.zeros((size, size), dtype=bool)
        self._asked_matrix = np.zeros((size, size), dtype=bool)
        # diagonals are known SAME
        np.fill_diagonal(self._v_matrix_working, True)
        np.fill_diagonal(self._h_matrix_working, True)
        np.fill_diagonal(self._asked_matrix, True)
        self._refresh_from_working()

    @staticmethod
    def _dir_to_val(dir: Dir, axis: Literal['vertical', 'horizontal']) -> int:
        """Convert a Dir enum to its matrix value representation in {-1,0,1}. UNKNOWN -> 0 but will be masked unknown by caller."""
        mapping = {
            'vertical': {Dir.SAME: 0, Dir.FORWARD: 1, Dir.BACKWARD: -1, Dir.UNKNOWN: 0},
            'horizontal': {Dir.SAME: 0, Dir.RIGHT: 1, Dir.LEFT: -1, Dir.UNKNOWN: 0}
        }
        return mapping[axis][dir]

    @staticmethod
    def _val_to_dir(value: int, axis: Literal['vertical', 'horizontal']) -> Dir:
        """Convert a matrix value to its Dir enum representation."""
        mapping = {
            'vertical': {0: Dir.SAME, 1: Dir.FORWARD, -1: Dir.BACKWARD},
            'horizontal': {0: Dir.SAME, 1: Dir.RIGHT, -1: Dir.LEFT}
        }
        return mapping[axis].get(int(value), Dir.UNKNOWN)

    def _init_from_objects(self, objects: List[Object]) -> None:
        """Initialize state from absolute coordinates of objects."""
        n = len(objects)
        self._v_matrix_working = np.zeros((n, n), dtype=bool)
        self._h_matrix_working = np.zeros((n, n), dtype=bool)
        # fill direct edges
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue
                rel = PairwiseRelationship.relationship(obj1.pos, obj2.pos, full=False)
                v_val = self._dir_to_val(rel.dir_pair.vert, 'vertical')
                h_val = self._dir_to_val(rel.dir_pair.horiz, 'horizontal')
                if v_val >= 0:
                    self._v_matrix_working[i, j] = True
                if v_val <= 0:
                    self._v_matrix_working[j, i] = True
                if h_val >= 0:
                    self._h_matrix_working[i, j] = True
                if h_val <= 0:
                    self._h_matrix_working[j, i] = True
        np.fill_diagonal(self._v_matrix_working, True)
        np.fill_diagonal(self._h_matrix_working, True)
        # asked: everything known initially
        self._asked_matrix = np.ones((n, n), dtype=bool)
        self._refresh_from_working()

    def _transitive_closure(self, W: np.ndarray) -> np.ndarray:
        """Floyd-Warshall style boolean transitive closure."""
        n = W.shape[0]
        for k in range(n):
            W |= (W[:, k][:, None] & W[k, :][None, :])
        return W

    def _refresh_from_working(self) -> None:
        """Recompute final matrices and known masks from working closure."""
        self._v_matrix_working = self._transitive_closure(self._v_matrix_working.copy())
        self._h_matrix_working = self._transitive_closure(self._h_matrix_working.copy())
        self._v_known = (self._v_matrix_working | self._v_matrix_working.T)
        self._h_known = (self._h_matrix_working | self._h_matrix_working.T)
        self._v_matrix = (self._v_matrix_working.astype(np.int8) - self._v_matrix_working.T.astype(np.int8)).astype(np.int8)
        self._h_matrix = (self._h_matrix_working.astype(np.int8) - self._h_matrix_working.T.astype(np.int8)).astype(np.int8)
        
    @staticmethod
    def create_graph_from_coordinates(coordinates: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a graph from a list of coordinates
        """
        
        # Initialize matrices
        _v_matrix = np.zeros((len(coordinates), len(coordinates)), dtype=np.int8)
        _h_matrix = np.zeros((len(coordinates), len(coordinates)), dtype=np.int8)
        
        # Fill matrices based on coordinates
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if i == j:
                    continue
                rel = PairwiseRelationship.relationship(coordinates[i], coordinates[j], full=False)
                _v_matrix[i, j] = DirectionalGraph._dir_to_val(rel.dir_pair.vert, 'vertical')
                _h_matrix[i, j] = DirectionalGraph._dir_to_val(rel.dir_pair.horiz, 'horizontal')
        
        return _v_matrix, _h_matrix
        
        
        # create a new graph with the same size as the original graph
        

    def add_edge(self, obj1: str, obj2: str, dir_pair: DirPair) -> bool:
        """
        Add a directional edge between two objects and update all inferrable relationships.
        Returns True if this was a novel query (at least one dimension previously unknown).
        """
        assert self.is_explore, "Cannot add edges when is_explore is False"
        i = self._ensure_index(obj1)
        j = self._ensure_index(obj2)
        v_val = self._dir_to_val(dir_pair.vert, 'vertical')
        h_val = self._dir_to_val(dir_pair.horiz, 'horizontal')
        if dir_pair.vert == Dir.UNKNOWN or dir_pair.horiz == Dir.UNKNOWN:
            raise ValueError("Direction must be fully known for add_edge; use add_partial_edge for unknown component")

        novel_query = (not self._v_known[i, j]) or (not self._h_known[i, j])

        # vertical
        if v_val > 0:
            self._v_matrix_working[i, j] = True
        elif v_val < 0:
            self._v_matrix_working[j, i] = True
        else:  # same
            self._v_matrix_working[i, j] = True
            self._v_matrix_working[j, i] = True

        # horizontal
        if h_val > 0:
            self._h_matrix_working[i, j] = True
        elif h_val < 0:
            self._h_matrix_working[j, i] = True
        else:
            self._h_matrix_working[i, j] = True
            self._h_matrix_working[j, i] = True

        # recompute closure and finals
        self._refresh_from_working()

        # mark asked
        self._asked_matrix[i, j] = True
        self._asked_matrix[j, i] = True
        return novel_query

    def add_partial_edge(self, obj1: str, obj2: str, dir_pair: DirPair) -> None:
        """
        Add a partial directional edge with at least one unknown direction.
        NOTE: the relationship is inferred
        
        Args:
            obj1: first object (name or index)
            obj2: second object (name or index)
            dir_pair: Direction pair with at least one unknown direction
        """
        assert self.is_explore, "Cannot add edges when is_explore is False"
        i = self._ensure_index(obj1)
        j = self._ensure_index(obj2)
        # vertical
        if dir_pair.vert != Dir.UNKNOWN:
            v_val = self._dir_to_val(dir_pair.vert, 'vertical')
            if v_val > 0:
                self._v_matrix_working[i, j] = True
            elif v_val < 0:
                self._v_matrix_working[j, i] = True
            else:
                self._v_matrix_working[i, j] = True
                self._v_matrix_working[j, i] = True
        # horizontal
        if dir_pair.horiz != Dir.UNKNOWN:
            h_val = self._dir_to_val(dir_pair.horiz, 'horizontal')
            if h_val > 0:
                self._h_matrix_working[i, j] = True
            elif h_val < 0:
                self._h_matrix_working[j, i] = True
            else:
                self._h_matrix_working[i, j] = True
                self._h_matrix_working[j, i] = True
        self._refresh_from_working()

    def add_node(self, new_name: str, anchor: str, dir_pair: DirPair) -> None:
        """Add a new object by name; its relation to anchor is given by dir_pair."""
        assert self.is_explore, "Cannot add node when is_explore is False"
        if new_name in self.name_to_idx:
            raise ValueError(f"Node '{new_name}' already exists")
        # append node
        self.nodes.append(new_name)
        self.name_to_idx[new_name] = len(self.nodes) - 1
        self.size = len(self.nodes)
        # expand matrices
        self._v_matrix_working = np.pad(self._v_matrix_working, ((0, 1), (0, 1)), constant_values=False)
        self._h_matrix_working = np.pad(self._h_matrix_working, ((0, 1), (0, 1)), constant_values=False)
        self._asked_matrix = np.pad(self._asked_matrix, ((0, 1), (0, 1)), constant_values=False)
        idx = self.name_to_idx[new_name]
        self._v_matrix_working[idx, idx] = True
        self._h_matrix_working[idx, idx] = True
        self._asked_matrix[idx, idx] = True
        # set relation
        self.add_edge(new_name, anchor, dir_pair)

    def move_node(self, obj1: str, obj2: str, dir_pair: DirPair) -> None:
        """
        Move an object (obj1) to a new position relative to another object (obj2).
        Simplified as setting the relation between them, then recomputing closure.
        """
        assert self.is_explore, "Cannot move node when is_explore is False"
        i = self._ensure_index(obj1)
        # clear relationships of obj1 to all others: become unknown
        self._v_matrix_working[i, :] = False
        self._v_matrix_working[:, i] = False
        self._h_matrix_working[i, :] = False
        self._h_matrix_working[:, i] = False
        # keep self relations
        self._v_matrix_working[i, i] = True
        self._h_matrix_working[i, i] = True
        # clear asked between obj1 and all others because position changed
        self._asked_matrix[i, :] = False
        self._asked_matrix[:, i] = False
        self._asked_matrix[i, i] = True
        # refresh known/finals, then set new relation
        self._refresh_from_working()
        self.add_edge(obj1, obj2, dir_pair)

        

    def rotate_axis(self, degree: int) -> None:
        """
        Rotate the coordinate system clockwise by the specified degree.
        
        Args:
            degree: Rotation angle (90, 180, or 270 degrees)
            
        Raises:
            ValueError: If degree is not valid or is_explore is False
        """
        assert self.is_explore, "Cannot rotate when is_explore is False"
        assert degree in self.VALID_ROTATION_DEGREES, f"Degree must be one of {self.VALID_ROTATION_DEGREES}"

        v_working = copy.deepcopy(self._v_matrix_working)
        h_working = copy.deepcopy(self._h_matrix_working)

        rotation_maps = {
            0: (v_working, h_working),
            90: (h_working, v_working.T),
            180: (v_working.T, h_working.T),
            270: (h_working.T, v_working)
        }
        
        self._v_matrix_working, self._h_matrix_working = rotation_maps[degree]
        self._refresh_from_working()

    def get_direction(self, obj1: str, obj2: str) -> DirPair:  # NOTE old name: get_dir_rel
        """
        Get the directional relationship between two objects, may be unknown
        
        Args:
            obj1: first object (name or index)
            obj2: second object (name or index)
            
        Returns:
            Direction Pair (DirPair)
        """
        i = self._ensure_index(obj1)
        j = self._ensure_index(obj2)
        if not self._v_known[i, j]:
            v_dir = Dir.UNKNOWN
        else:
            v_dir = self._val_to_dir(int(self._v_matrix[i, j]), 'vertical')
        if not self._h_known[i, j]:
            h_dir = Dir.UNKNOWN
        else:
            h_dir = self._val_to_dir(int(self._h_matrix[i, j]), 'horizontal')
        return DirPair(horiz=h_dir, vert=v_dir)
    
    def get_unknown_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all pairs of objects with unknown spatial relationships.
        
        Returns:
            List of tuples containing (obj1_id, obj2_id) where at least one
            dimension (vertical or horizontal) of their relationship is unknown.
            Only returns unique pairs (no duplicates or self-pairs).
        """
        unknown_masks = ~(self._v_known & self._h_known)
        unknown_masks = unknown_masks[:self.size, :self.size]
        
        # Use numpy to find indices where relationships are unknown
        unknown_indices = np.where(np.triu(unknown_masks, k=1))
        unknown_pairs = [(self.nodes[i.item()], self.nodes[j.item()]) for i,j in zip(unknown_indices[0], unknown_indices[1])]
        
        return unknown_pairs
    
    def get_inferable_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all pairs of objects with known spatial relationships.
        
        Returns:
            List of tuples containing (obj1_id, obj2_id) where both
        """
        known_masks = (self._v_known & self._h_known)
        known_masks = known_masks[:self.size, :self.size]
        inferable_masks = known_masks & ~self._asked_matrix
        inferable_indices = np.where(np.triu(inferable_masks, k=1))
        inferable_pairs = [(self.nodes[i.item()], self.nodes[j.item()]) for i,j in zip(inferable_indices[0], inferable_indices[1])]
        
        return inferable_pairs
        
        
    def to_dict(self) -> Dict[str, List[List[int]]]:
        """Serialize graph including nodes, matrices, and masks."""
        return {
            'nodes': self.nodes,
            'v_matrix': self._v_matrix.astype(int).tolist(),
            'h_matrix': self._h_matrix.astype(int).tolist(),
            'v_known': self._v_known.astype(bool).tolist(),
            'h_known': self._h_known.astype(bool).tolist(),
            'v_matrix_working': self._v_matrix_working.astype(bool).tolist(),
            'h_matrix_working': self._h_matrix_working.astype(bool).tolist(),
            'asked_matrix': self._asked_matrix.astype(bool).tolist(),
            'size': self.size,
            'is_explore': self.is_explore,
        }
    
    @classmethod
    def from_dict(cls, graph_dict: Dict[str, List[List[int]]]) -> 'DirectionalGraph':
        """Deserialize DirectionalGraph including nodes and masks."""
        instance = cls(objects=[], is_explore=graph_dict.get('is_explore', True))
        instance.nodes = list(graph_dict.get('nodes', []))
        instance.name_to_idx = {name: i for i, name in enumerate(instance.nodes)}
        instance.size = graph_dict.get('size', len(instance.nodes))
        instance._v_matrix = np.array(graph_dict['v_matrix'], dtype=np.int8)
        instance._h_matrix = np.array(graph_dict['h_matrix'], dtype=np.int8)
        instance._v_known = np.array(graph_dict['v_known'], dtype=bool)
        instance._h_known = np.array(graph_dict['h_known'], dtype=bool)
        instance._v_matrix_working = np.array(graph_dict['v_matrix_working'], dtype=bool)
        instance._h_matrix_working = np.array(graph_dict['h_matrix_working'], dtype=bool)
        instance._asked_matrix = np.array(graph_dict['asked_matrix'], dtype=bool)
        return instance
    
    def copy(self) -> 'DirectionalGraph':
        """Create a deep copy of the graph, including nodes and matrices."""
        new_graph = object.__new__(DirectionalGraph)
        new_graph.nodes = list(self.nodes)
        new_graph.name_to_idx = dict(self.name_to_idx)
        new_graph.size = self.size
        new_graph.is_explore = self.is_explore
        new_graph._v_matrix = copy.deepcopy(self._v_matrix)
        new_graph._h_matrix = copy.deepcopy(self._h_matrix)
        new_graph._v_matrix_working = copy.deepcopy(self._v_matrix_working)
        new_graph._h_matrix_working = copy.deepcopy(self._h_matrix_working)
        new_graph._v_known = copy.deepcopy(self._v_known)
        new_graph._h_known = copy.deepcopy(self._h_known)
        new_graph._asked_matrix = copy.deepcopy(self._asked_matrix)
        return new_graph

    # ----- node utilities -----
    def _ensure_index(self, name: str) -> int:
        if name not in self.name_to_idx:
            raise ValueError(f"Object '{name}' not found in graph")
        return self.name_to_idx[name]

    def remove_node(self, name: str) -> None:
        """Remove a node by name and shrink matrices accordingly."""
        idx = self._ensure_index(name)
        # remove from nodes
        removed_name = self.nodes.pop(idx)
        del self.name_to_idx[removed_name]
        # shift indices
        for k, v in list(self.name_to_idx.items()):
            if v > idx:
                self.name_to_idx[k] = v - 1
        # shrink matrices
        for attr in ['_v_matrix', '_h_matrix', '_v_matrix_working', '_h_matrix_working', '_v_known', '_h_known', '_asked_matrix']:
            mat = getattr(self, attr)
            mat = np.delete(mat, idx, axis=0)
            mat = np.delete(mat, idx, axis=1)
            setattr(self, attr, mat)
        self.size = len(self.nodes)

    def rename_node(self, old_name: str, new_name: str) -> None:
        if new_name in self.name_to_idx:
            raise ValueError(f"Node '{new_name}' already exists")
        idx = self._ensure_index(old_name)
        self.nodes[idx] = new_name
        del self.name_to_idx[old_name]
        self.name_to_idx[new_name] = idx
