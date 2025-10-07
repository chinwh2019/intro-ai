"""
State representation for search problems
"""

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class State:
    """Represents a state in the search space"""

    position: Tuple[int, int]  # (row, col)

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"State({self.position})"


@dataclass
class Node:
    """Node in the search tree"""

    state: State
    parent: Optional['Node'] = None
    action: Optional[str] = None
    path_cost: float = 0.0
    heuristic_cost: float = 0.0

    def __lt__(self, other):
        """For priority queue comparison"""
        return self.total_cost() < other.total_cost()

    def total_cost(self) -> float:
        """Total cost (f = g + h)"""
        return self.path_cost + self.heuristic_cost

    def get_path(self) -> list:
        """Reconstruct path from root to this node"""
        path = []
        node = self
        while node:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))
