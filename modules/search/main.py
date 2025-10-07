"""
Main application for Search Algorithms Module
"""

import pygame
import sys
import time
from typing import Optional
from modules.search.config import config
from modules.search.core.environment import Maze
from modules.search.core.base_algorithm import SearchAlgorithm
from modules.search.algorithms.bfs import BFS
from modules.search.algorithms.dfs import DFS
from modules.search.algorithms.ucs import UCS
from modules.search.algorithms.astar import AStar
from modules.search.algorithms.greedy import GreedyBestFirst
from modules.search.ui.visualizer import SearchVisualizer

class SearchApp:
    """Main application for search algorithms"""

    def __init__(self):
        self.maze = Maze(config.MAZE_WIDTH, config.MAZE_HEIGHT, config.MAZE_COMPLEXITY)
        self.visualizer = SearchVisualizer(self.maze)

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
        print("  Q: Quit")
        print("=" * 50)

    def select_algorithm(self, key: str):
        """Select and start algorithm"""
        if key in self.algorithms:
            name, algo_class = self.algorithms[key]
            print(f"\nStarting {name}...")

            # Create new algorithm instance
            self.current_algorithm = algo_class(self.maze)
            self.search_generator = self.current_algorithm.search()
            self.algorithm_complete = False
            self.paused = False

            print(f"Maze size: {self.maze.width}x{self.maze.height}")
            print(f"Start: {self.maze.start}, Goal: {self.maze.goal}")

    def reset_maze(self):
        """Reset maze and algorithm"""
        print("\nGenerating new maze...")
        self.maze = Maze(config.MAZE_WIDTH, config.MAZE_HEIGHT, config.MAZE_COMPLEXITY)
        self.visualizer = SearchVisualizer(self.maze)
        self.current_algorithm = None
        self.search_generator = None
        self.algorithm_complete = False
        print("Maze reset. Select an algorithm to start.")

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
