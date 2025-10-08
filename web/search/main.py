"""
Search Algorithms Module - Web Version
Runs in browser via Pygbag/WebAssembly

Controls:
  1-5: Select algorithm (BFS, DFS, UCS, A*, Greedy)
  SPACE: Pause/Resume
  S: Step through (when paused)
  R: Reset maze
  T: Toggle random start/goal
  Q: Quit
"""

# /// script
# [pygbag]
# autorun = true
# ///

import asyncio
import pygame
import sys
from typing import Optional

# Import from local modules (no 'modules.search' prefix for web)
import config
from core.environment import Maze
from core.base_algorithm import SearchAlgorithm
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.ucs import UCS
from algorithms.astar import AStar
from algorithms.greedy import GreedyBestFirst
from ui.visualizer import SearchVisualizer


class SearchApp:
    """Web-compatible search application"""

    def __init__(self):
        self.maze = self._create_maze()
        self.visualizer = SearchVisualizer(self.maze)

        # Available algorithms (use key codes for web compatibility)
        self.algorithms = {
            pygame.K_1: ('BFS', BFS),
            pygame.K_2: ('DFS', DFS),
            pygame.K_3: ('UCS', UCS),
            pygame.K_4: ('A*', AStar),
            pygame.K_5: ('Greedy', GreedyBestFirst),
        }

        # Current algorithm
        self.current_algorithm: Optional[SearchAlgorithm] = None
        self.search_generator = None

        # Control state
        self.running = True
        self.paused = False
        self.step_mode = False
        self.algorithm_complete = False

        # For web: time-based stepping (more consistent across devices)
        self.last_step_time = 0
        self.step_interval = config.config.STEP_DELAY / config.config.ANIMATION_SPEED

        print("Search Algorithms - Web Version")
        print("=" * 50)
        print("Controls:")
        print("  1-5: Select algorithm")
        print("  SPACE: Pause/Resume")
        print("  S: Step")
        print("  R: Reset maze")
        print("  T: Toggle random start/goal")
        print("  Q: Quit")
        print("=" * 50)

    def _create_maze(self) -> Maze:
        """Create maze using current config settings"""
        return Maze(
            width=config.config.MAZE_WIDTH,
            height=config.config.MAZE_HEIGHT,
            complexity=config.config.MAZE_COMPLEXITY,
            start_pos=config.config.START_POSITION,
            goal_pos=config.config.GOAL_POSITION,
            random_start_goal=config.config.RANDOM_START_GOAL
        )

    def select_algorithm(self, key):
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
        self.maze = self._create_maze()
        self.visualizer = SearchVisualizer(self.maze)
        self.current_algorithm = None
        self.search_generator = None
        self.algorithm_complete = False
        print(f"Maze reset ({config.config.MAZE_WIDTH}x{config.config.MAZE_HEIGHT})")
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
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                # Algorithm selection
                if event.key in self.algorithms:
                    self.select_algorithm(event.key)

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
                    config.config.RANDOM_START_GOAL = not config.config.RANDOM_START_GOAL
                    mode = "ON" if config.config.RANDOM_START_GOAL else "OFF"
                    print(f"\nRandom start/goal: {mode}")
                    print("Press R to generate new maze with this setting")

                elif event.key == pygame.K_q:
                    self.running = False

    def update(self, current_time: float):
        """Update application state (web-compatible time-based)"""
        if not self.paused and not self.algorithm_complete:
            # Time-based stepping for consistent speed across devices
            if current_time - self.last_step_time >= self.step_interval:
                self.step_algorithm()
                self.last_step_time = current_time

        elif self.step_mode:
            self.step_algorithm()
            self.step_mode = False

    def render(self):
        """Render application"""
        self.visualizer.render()


async def main():
    """
    Async main loop for web compatibility

    CRITICAL: The 'async' and 'await asyncio.sleep(0)' are REQUIRED
    for pygbag to work correctly in the browser!
    """
    print("Loading Search Algorithms...")

    # Initialize application
    app = SearchApp()
    clock = pygame.time.Clock()

    print("✓ Ready! Press 1-5 to select an algorithm")

    # Main game loop (MUST be async for web)
    while app.running:
        # Get current time in seconds
        current_time = pygame.time.get_ticks() / 1000.0

        # Process events and update
        app.handle_events()
        app.update(current_time)
        app.render()

        # CRITICAL: Yield control to browser event loop
        # Without this, the browser will freeze!
        await asyncio.sleep(0)

        # Frame rate limiting
        clock.tick(config.config.FPS)

    pygame.quit()
    print("Thank you for using Search Algorithms!")


# Entry point
if __name__ == '__main__':
    # Use asyncio to run async main function
    asyncio.run(main())
