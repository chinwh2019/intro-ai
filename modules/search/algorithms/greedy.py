"""
Greedy Best-First Search implementation
"""

import heapq
from typing import Generator
from modules.search.core.base_algorithm import SearchAlgorithm
from modules.search.core.state import State, Node

class GreedyBestFirst(SearchAlgorithm):
    """Greedy Best-First Search"""

    def __init__(self, maze, heuristic: str = 'manhattan'):
        super().__init__(maze)

        # Set heuristic function
        if heuristic == 'manhattan':
            self.heuristic_func = maze.manhattan_distance
        elif heuristic == 'euclidean':
            self.heuristic_func = maze.euclidean_distance
        else:
            self.heuristic_func = maze.manhattan_distance

    def search(self) -> Generator[dict, None, None]:
        """Execute Greedy Best-First Search"""
        # Initialize frontier with start node
        h_start = self.heuristic_func(self.start_state.position, self.goal_state.position)
        start_node = Node(
            state=self.start_state,
            path_cost=0,
            heuristic_cost=h_start
        )

        frontier = [(h_start, 0, start_node)]  # (h_cost, tie_breaker, node)
        self.frontier_states = {self.start_state}

        # Track reached states
        reached = {self.start_state}
        counter = 0

        while frontier:
            # Get node with lowest heuristic cost
            h_cost, _, self.current_node = heapq.heappop(frontier)
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
                    h_cost = self.heuristic_func(neighbor_pos, self.goal_state.position)
                    child_node = Node(
                        state=neighbor_state,
                        parent=self.current_node,
                        action=self.maze.get_action_name(
                            self.current_node.state.position,
                            neighbor_pos
                        ),
                        path_cost=self.current_node.path_cost + 1,
                        heuristic_cost=h_cost
                    )
                    counter += 1
                    heapq.heappush(frontier, (h_cost, counter, child_node))
                    self.frontier_states.add(neighbor_state)
                    self.nodes_generated += 1

            # Update max frontier size
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))

        # No solution found
        yield self.get_current_visualization_state()
