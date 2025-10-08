"""
Main application for Search Algorithms Module
"""

import pygame
import sys
import time
from typing import Optional
from config import config
from core.environment import Maze
from core.base_algorithm import SearchAlgorithm
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.ucs import UCS
from algorithms.astar import AStar
from algorithms.greedy import GreedyBestFirst
from ui.visualizer import SearchVisualizer

class SearchApp:
    """Main application for search algorithms"""

    def __init__(self):
        self.maze = self._create_maze()
        self.visualizer = SearchVisualizer(self.maze, on_parameter_change=self.on_parameter_change)

        # Available algorithms
        self.algorithms = {
            '1': ('BFS', BFS),
            '2': ('DFS', DFS),
            '3': ('UCS', UCS),
            '4': ('A*', AStar),
            '5': ('Greedy', GreedyBestFirst),
        }

        # Current algorithm
        self.current_algorithm: Optional[SearchAlgorithm] = None
        self.search_generator = None
        self.current_algorithm_key = None  # Track which algorithm is running

        # Interactive parameters
        self.heuristic_weight = 1.0  # Current heuristic weight for A*/Greedy

        # Control state
        self.running = True
        self.paused = False
        self.step_mode = False
        self.algorithm_complete = False

        print("Search Algorithms Visualization")
        print("=" * 50)
        print("Controls:")
        print("  1-5: Select algorithm")
        print("  SPACE: Pause/Resume")
        print("  S: Step (when paused)")
        print("  R: Reset maze")
        print("  T: Toggle random start/goal")
        print("  Q: Quit")
        print("=" * 50)
        print(f"Random start/goal: {'ON' if config.RANDOM_START_GOAL else 'OFF'}")
        print("=" * 50)

    def _create_maze(self) -> Maze:
        """Create maze using current config settings"""
        return Maze(
            width=config.MAZE_WIDTH,
            height=config.MAZE_HEIGHT,
            complexity=config.MAZE_COMPLEXITY,
            start_pos=config.START_POSITION,
            goal_pos=config.GOAL_POSITION,
            random_start_goal=config.RANDOM_START_GOAL
        )

    def on_parameter_change(self, params: dict):
        """Handle parameter changes from sliders"""
        try:
            print("\n" + "=" * 50)
            print("🔄 Applying parameters...")
            print(f"  Speed: {params['speed']:.1f}x")
            print(f"  Heuristic weight: {params['heuristic_weight']:.2f}")
            print(f"  Complexity: {params['complexity']:.2f}")

            # Apply speed immediately (affects current search)
            config.ANIMATION_SPEED = params['speed']

            # Store heuristic weight (applies to next algorithm start)
            self.heuristic_weight = params['heuristic_weight']

            # Store complexity (applies on next reset)
            config.MAZE_COMPLEXITY = params['complexity']

            # If A* or Greedy is running, restart with new weight
            if self.current_algorithm_key in ['4', '5'] and not self.algorithm_complete:
                print(f"  Restarting {self.algorithms[self.current_algorithm_key][0]} with new weight...")
                self.select_algorithm(self.current_algorithm_key)

            print("✓ Parameters applied!")
            print("=" * 50)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    def select_algorithm(self, key: str):
        """Select and start algorithm"""
        if key in self.algorithms:
            name, algo_class = self.algorithms[key]
            print(f"\nStarting {name}...")

            # Create new algorithm instance with heuristic weight for A*/Greedy
            if key in ['4', '5']:  # A* or Greedy
                # Create custom heuristic with weight
                def weighted_heuristic(pos1, pos2):
                    manhattan = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    return self.heuristic_weight * manhattan

                self.current_algorithm = algo_class(self.maze, heuristic_func=weighted_heuristic)
                print(f"  Using heuristic weight: {self.heuristic_weight:.2f}")
                if self.heuristic_weight > 1.0:
                    print(f"  ⚠ Inadmissible - may not find optimal solution")
                elif self.heuristic_weight == 1.0:
                    print(f"  ✓ Admissible - optimal solution guaranteed")
            else:
                self.current_algorithm = algo_class(self.maze)

            self.search_generator = self.current_algorithm.search()
            self.algorithm_complete = False
            self.paused = False
            self.current_algorithm_key = key

            # Update visualizer to show current algorithm
            self.visualizer.set_algorithm(name)

            print(f"Maze size: {self.maze.width}x{self.maze.height}")
            print(f"Start: {self.maze.start}, Goal: {self.maze.goal}")

    def reset_maze(self):
        """Reset maze and algorithm"""
        print("\nGenerating new maze...")
        self.maze = self._create_maze()
        self.visualizer = SearchVisualizer(self.maze, on_parameter_change=self.on_parameter_change)
        self.current_algorithm = None
        self.search_generator = None
        self.current_algorithm_key = None
        self.algorithm_complete = False

        # Reset algorithm display
        self.visualizer.set_algorithm("None")

        print(f"Maze reset ({config.MAZE_WIDTH}x{config.MAZE_HEIGHT}, complexity={config.MAZE_COMPLEXITY:.2f})")
        print(f"Start: {self.maze.start}, Goal: {self.maze.goal}")

    def step_algorithm(self):
        """Execute one step of the algorithm"""
        if self.search_generator and not self.algorithm_complete:
            try:
                viz_state = next(self.search_generator)
                self.visualizer.update_state(viz_state)

                # Check if solution found
                if viz_state.get('solution_found'):
                    self.algorithm_complete = True
                    stats = viz_state.get('stats', {})
                    print(f"\n✓ Solution found!")
                    print(f"  Path length: {stats.get('solution_length', 0)}")
                    print(f"  Nodes expanded: {stats.get('nodes_expanded', 0)}")
                    print(f"  Nodes generated: {stats.get('nodes_generated', 0)}")
            except StopIteration:
                self.algorithm_complete = True
                print("\nSearch complete (no solution found)")

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            # Pass event to parameter panel first (for slider/button handling)
            self.visualizer.handle_parameter_event(event)

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                # Algorithm selection
                if event.unicode in self.algorithms:
                    self.select_algorithm(event.unicode)

                # Controls
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")

                elif event.key == pygame.K_s:
                    if self.paused:
                        self.step_mode = True

                elif event.key == pygame.K_r:
                    self.reset_maze()

                elif event.key == pygame.K_t:
                    # Toggle random start/goal mode
                    config.RANDOM_START_GOAL = not config.RANDOM_START_GOAL
                    mode = "ON" if config.RANDOM_START_GOAL else "OFF"
                    print(f"\nRandom start/goal: {mode}")
                    print("Press R to generate new maze with this setting")

                elif event.key == pygame.K_q:
                    self.running = False

    def update(self):
        """Update application state"""
        if not self.paused and not self.algorithm_complete:
            # Execute algorithm step
            self.step_algorithm()

            # Delay based on speed setting
            time.sleep(config.STEP_DELAY / config.ANIMATION_SPEED)

        elif self.step_mode:
            self.step_algorithm()
            self.step_mode = False

    def render(self):
        """Render application"""
        self.visualizer.render()

    def run(self):
        """Main application loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()

        pygame.quit()
        sys.exit()


def main():
    """Entry point"""
    app = SearchApp()
    app.run()


if __name__ == '__main__':
    main()
