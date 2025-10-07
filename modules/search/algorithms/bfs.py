"""
Breadth-First Search implementation
"""

from collections import deque
from typing import Generator
from modules.search.core.base_algorithm import SearchAlgorithm
from modules.search.core.state import State, Node

class BFS(SearchAlgorithm):
    """Breadth-First Search"""

    def search(self) -> Generator[dict, None, None]:
        """Execute BFS"""
        # Initialize frontier with start node
        start_node = Node(state=self.start_state, path_cost=0)
        frontier = deque([start_node])
        self.frontier_states = {self.start_state}

        # Track reached states to avoid revisiting
        reached = {self.start_state}

        while frontier:
            # Get next node from frontier (FIFO)
            self.current_node = frontier.popleft()
            self.frontier_states.discard(self.current_node.state)

            # Add to explored
            self.explored.add(self.current_node.state)
            self.nodes_expanded += 1
            self.step_count += 1

            # Yield current state for visualization
            yield self.get_current_visualization_state()

            # Check if goal
            if self.is_goal(self.current_node.state):
                self.solution_found = True
                self.solution_path = self.current_node.get_path()
                yield self.get_current_visualization_state()
                return

            # Expand neighbors
            neighbors = self.maze.get_neighbors(self.current_node.state.position)
            for neighbor_pos in neighbors:
                neighbor_state = State(neighbor_pos)

                if neighbor_state not in reached:
                    reached.add(neighbor_state)
                    child_node = Node(
                        state=neighbor_state,
                        parent=self.current_node,
                        action=self.maze.get_action_name(
                            self.current_node.state.position,
                            neighbor_pos
                        ),
                        path_cost=self.current_node.path_cost + 1
                    )
                    frontier.append(child_node)
                    self.frontier_states.add(neighbor_state)
                    self.nodes_generated += 1

            # Update max frontier size
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))

        # No solution found
        yield self.get_current_visualization_state()
