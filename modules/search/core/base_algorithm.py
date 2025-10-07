"""
Base class for search algorithms
"""

from abc import ABC, abstractmethod
from typing import List, Set, Optional, Generator
from modules.search.core.state import State, Node
from modules.search.core.environment import Maze

class SearchAlgorithm(ABC):
    """Base class for all search algorithms"""

    def __init__(self, maze: Maze):
        self.maze = maze
        self.start_state = State(maze.start)
        self.goal_state = State(maze.goal)

        # Statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0

        # State tracking
        self.explored: Set[State] = set()
        self.frontier_states: Set[State] = set()
        self.current_node: Optional[Node] = None
        self.solution_path: List[State] = []
        self.solution_found = False

        # For visualization
        self.step_count = 0

    @abstractmethod
    def search(self) -> Generator[dict, None, None]:
        """
        Execute search algorithm (generator for step-by-step visualization)

        Yields:
            dict: Current state of the search with keys:
                - 'explored': Set of explored states
                - 'frontier': Set of frontier states
                - 'current': Current node being expanded
                - 'path': Current path (if solution found)
                - 'solution_found': Boolean
                - 'stats': Dictionary of statistics
        """
        pass

    def get_statistics(self) -> dict:
        """Get search statistics"""
        return {
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'max_frontier_size': self.max_frontier_size,
            'solution_length': len(self.solution_path) if self.solution_found else 0,
            'steps': self.step_count,
        }

    def is_goal(self, state: State) -> bool:
        """Check if state is goal"""
        return state == self.goal_state

    def get_current_visualization_state(self) -> dict:
        """Get current state for visualization"""
        return {
            'explored': self.explored.copy(),
            'frontier': self.frontier_states.copy(),
            'current': self.current_node.state if self.current_node else None,
            'path': self.solution_path.copy(),
            'solution_found': self.solution_found,
            'stats': self.get_statistics(),
        }
