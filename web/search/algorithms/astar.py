"""
A* Search implementation
"""

import heapq
from typing import Generator, Callable, Union
from core.base_algorithm import SearchAlgorithm
from core.state import State, Node

class AStar(SearchAlgorithm):
    """A* Search with pluggable heuristic"""

    def __init__(
        self,
        maze,
        heuristic: Union[str, Callable] = 'manhattan',
        heuristic_func: Callable = None
    ):
        super().__init__(maze)

        # Set heuristic function
        # Priority: heuristic_func > heuristic (string) > default
        if heuristic_func:
            # Custom function provided directly
            self.heuristic_func = heuristic_func
            self.heuristic_name = 'custom'
        elif callable(heuristic):
            # Heuristic is already a function
            self.heuristic_func = heuristic
            self.heuristic_name = 'custom'
        elif isinstance(heuristic, str):
            # String name - use built-in or from heuristics module
            if heuristic == 'manhattan':
                self.heuristic_func = maze.manhattan_distance
                self.heuristic_name = 'manhattan'
            elif heuristic == 'euclidean':
                self.heuristic_func = maze.euclidean_distance
                self.heuristic_name = 'euclidean'
            else:
                # Try to get from heuristics module
                try:
                    from heuristics import get_heuristic
                    self.heuristic_func = get_heuristic(heuristic)
                    self.heuristic_name = heuristic
                except:
                    self.heuristic_func = maze.manhattan_distance
                    self.heuristic_name = 'manhattan'
        else:
            # Default
            self.heuristic_func = maze.manhattan_distance
            self.heuristic_name = 'manhattan'

    def search(self) -> Generator[dict, None, None]:
        """Execute A* search"""
        # Initialize frontier with start node
        h_start = self.heuristic_func(self.start_state.position, self.goal_state.position)
        start_node = Node(
            state=self.start_state,
            path_cost=0,
            heuristic_cost=h_start
        )

        frontier = [(h_start, 0, start_node)]  # (f_cost, tie_breaker, node)
        self.frontier_states = {self.start_state}

        # Track best f-cost to reach each state
        best_f_cost = {self.start_state: h_start}
        counter = 0

        while frontier:
            # Get node with lowest f-cost
            f_cost, _, self.current_node = heapq.heappop(frontier)
            self.frontier_states.discard(self.current_node.state)

            # Skip if we've explored this state
            if self.current_node.state in self.explored:
                continue

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

                if neighbor_state not in self.explored:
                    # Calculate costs
                    g_cost = self.current_node.path_cost + 1
                    h_cost = self.heuristic_func(neighbor_pos, self.goal_state.position)
                    f_cost = g_cost + h_cost

                    # Add if better than previous f-cost
                    if f_cost < best_f_cost.get(neighbor_state, float('inf')):
                        best_f_cost[neighbor_state] = f_cost
                        child_node = Node(
                            state=neighbor_state,
                            parent=self.current_node,
                            action=self.maze.get_action_name(
                                self.current_node.state.position,
                                neighbor_pos
                            ),
                            path_cost=g_cost,
                            heuristic_cost=h_cost
                        )
                        counter += 1
                        heapq.heappush(frontier, (f_cost, counter, child_node))
                        self.frontier_states.add(neighbor_state)
                        self.nodes_generated += 1

            # Update max frontier size
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))

        # No solution found
        yield self.get_current_visualization_state()
