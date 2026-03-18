from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class BaseCogMetrics:
    overall: float = 0.0
    valid: bool = True

    def to_dict(self) -> Dict[str, float]:
        return {"overall": float(self.overall)}

    @classmethod
    def invalid(cls):
        return cls(overall=0.0, valid=False)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'BaseCogMetrics':
        if not isinstance(d, dict) or not d:
            return BaseCogMetrics.invalid()
        return BaseCogMetrics(
            overall=float(d.get('overall', 0.0)),
            valid=True,
        )
    @staticmethod
    def average(items: List['BaseCogMetrics']) -> 'BaseCogMetrics':
        valid_items = [i for i in items if isinstance(i, BaseCogMetrics) and i.valid]
        if not valid_items:
            return BaseCogMetrics.invalid()
        return BaseCogMetrics(
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )
    
@dataclass
class MapCogMetrics(BaseCogMetrics):
    dir: float = 0.0
    facing: float = 0.0
    pos: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "dir": float(self.dir),
            "facing": float(self.facing),
            "pos": float(self.pos),
            "overall": float(self.overall),
        }

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'MapCogMetrics':
        if not isinstance(d, dict) or not d:
            return MapCogMetrics.invalid()
        return MapCogMetrics(
            dir=float(d.get('dir', 0.0)),
            facing=float(d.get('facing', 0.0)),
            pos=float(d.get('pos', 0.0)),
            overall=float(d.get('overall', 0.0)),
            valid=True,
        )

    def __add__(self, other: 'MapCogMetrics') -> 'MapCogMetrics':
        return MapCogMetrics(
            dir=self.dir + other.dir,
            facing=self.facing + other.facing,
            pos=self.pos + other.pos,
            overall=self.overall + other.overall,
            valid=self.valid and other.valid,
        )

    def __truediv__(self, scalar: float) -> 'MapCogMetrics':
        if scalar == 0:
            return MapCogMetrics.invalid()
        return MapCogMetrics(
            dir=self.dir / scalar,
            facing=self.facing / scalar,
            pos=self.pos / scalar,
            overall=self.overall / scalar,
            valid=self.valid,
        )

    @staticmethod
    def average(items: List['MapCogMetrics']) -> 'MapCogMetrics':
        valid_items = [i for i in items if isinstance(i, MapCogMetrics) and i.valid]
        if not valid_items:
            return MapCogMetrics.invalid()
        return MapCogMetrics(
            dir=float(np.mean([i.dir for i in valid_items])),
            facing=float(np.mean([i.facing for i in valid_items])),
            pos=float(np.mean([i.pos for i in valid_items])),
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )

    @classmethod
    def invalid(cls) -> 'MapCogMetrics':
        return cls(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=False)


@dataclass
class AccuracyMetrics(BaseCogMetrics):
    """Binary/accuracy-style metric with validity flag.

    Uses 'overall' as the accuracy score to stay compatible with existing code.
    """

    def to_dict(self) -> Dict[str, float]:
        return {"overall": float(self.overall)}

    @classmethod
    def invalid(cls) -> 'AccuracyMetrics':
        return cls(overall=0.0, valid=False)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'AccuracyMetrics':
        if not isinstance(d, dict) or not d:
            return AccuracyMetrics.invalid()
        # Accept either 'overall' or 'acc' keys
        val = d.get('overall', d.get('acc', 0.0))
        return AccuracyMetrics(overall=float(val), valid=True)

@dataclass
class RelationMetrics(BaseCogMetrics):
    dir: float = 0.0
    dist: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"dir": float(self.dir), "dist": float(self.dist), "overall": float(self.overall)}

    @classmethod
    def invalid(cls) -> 'RelationMetrics':
        return cls(dir=0.0, dist=0.0, overall=0.0, valid=False)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'RelationMetrics':
        if not isinstance(d, dict) or not d:
            return RelationMetrics.invalid()
        return RelationMetrics(
            dir=float(d.get('dir', 0.0)),
            dist=float(d.get('dist', 0.0)),
            overall=float(d.get('overall', 0.0)),
            valid=True,
        )

    @staticmethod
    def average(items: List['RelationMetrics']) -> 'RelationMetrics':
        valid_items = [i for i in items if isinstance(i, RelationMetrics) and i.valid]
        if not valid_items:
            return RelationMetrics.invalid()
        import numpy as np
        return RelationMetrics(
            dir=float(np.mean([i.dir for i in valid_items])),
            dist=float(np.mean([i.dist for i in valid_items])),
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )

    def __add__(self, other: 'RelationMetrics') -> 'RelationMetrics':
        return RelationMetrics(
            dir=self.dir + other.dir,
            dist=self.dist + other.dist,
            overall=self.overall + other.overall,
            valid=self.valid and other.valid,
        )

    def __truediv__(self, scalar: float) -> 'RelationMetrics':
        if scalar == 0:
            return RelationMetrics.invalid()
        return RelationMetrics(
            dir=self.dir / scalar,
            dist=self.dist / scalar,
            overall=self.overall / scalar,
            valid=self.valid,
        )


@dataclass
class ConsistencySummary:
    local_vs_global: Optional[MapCogMetrics] = None
    rooms_vs_global_avg: Optional[MapCogMetrics] = None
    rooms_vs_global_per_room: Dict[str, MapCogMetrics] = field(default_factory=dict)
    map_vs_relations: Optional[float] = None
    relations_consistency: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "local_vs_global": (self.local_vs_global.to_dict() if self.local_vs_global and self.local_vs_global.valid else {}),
            "rooms_vs_global": {
                "average": (self.rooms_vs_global_avg.to_dict() if self.rooms_vs_global_avg and self.rooms_vs_global_avg.valid else {}),
                "per_room": {k: v.to_dict() for k, v in self.rooms_vs_global_per_room.items()},
            },
            "map_vs_relations": (float(self.map_vs_relations) if isinstance(self.map_vs_relations, (int, float)) else None),
            "relations_consistency": (float(self.relations_consistency) if isinstance(self.relations_consistency, (int, float)) else None),
        }


__all__ = [
    "BaseCogMetrics",
    "MapCogMetrics",
    "RelationMetrics",
    "AccuracyMetrics",
    "ConsistencySummary",
]


