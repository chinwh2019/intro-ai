"""
Uniform Cost Search implementation
"""

import heapq
from typing import Generator
from modules.search.core.base_algorithm import SearchAlgorithm
from modules.search.core.state import State, Node

class UCS(SearchAlgorithm):
    """Uniform Cost Search (Dijkstra's algorithm)"""

    def search(self) -> Generator[dict, None, None]:
        """Execute UCS"""
        # Initialize frontier with start node (priority queue)
        start_node = Node(state=self.start_state, path_cost=0)
        frontier = [(0, 0, start_node)]  # (cost, tie_breaker, node)
        self.frontier_states = {self.start_state}

        # Track best cost to reach each state
        best_cost = {self.start_state: 0}
        counter = 0  # Tie-breaker for heap

        while frontier:
            # Get node with lowest cost
            cost, _, self.current_node = heapq.heappop(frontier)
            self.frontier_states.discard(self.current_node.state)

            # Skip if we've found a better path to this state
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

                # Calculate cost (uniform cost of 1 per step)
                new_cost = self.current_node.path_cost + 1

                # Add if not explored and better than previous cost
                if (neighbor_state not in self.explored and
                    new_cost < best_cost.get(neighbor_state, float('inf'))):

                    best_cost[neighbor_state] = new_cost
                    child_node = Node(
                        state=neighbor_state,
                        parent=self.current_node,
                        action=self.maze.get_action_name(
                            self.current_node.state.position,
                            neighbor_pos
                        ),
                        path_cost=new_cost
                    )
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, child_node))
                    self.frontier_states.add(neighbor_state)
                    self.nodes_generated += 1

            # Update max frontier size
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))

        # No solution found
        yield self.get_current_visualization_state()
