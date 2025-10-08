"""Main application for MDP module"""

import pygame
import sys
import time
from modules.mdp.config import config
from modules.mdp.environments.grid_world import GridWorld
from modules.mdp.core.solver import ValueIteration, PolicyIteration
from modules.mdp.ui.visualizer import MDPVisualizer


class MDPApp:
    """Main MDP application"""

    def __init__(self):
        # Create environment
        self.grid_world = GridWorld(
            grid_size=config.GRID_SIZE,
            num_obstacles=config.NUM_OBSTACLES,
            noise=config.NOISE,
            discount=config.DISCOUNT,
            living_reward=config.LIVING_REWARD
        )

        # Create visualizer
        self.visualizer = MDPVisualizer(self.grid_world)

        # Create solver (default: Value Iteration)
        self.solver = ValueIteration(
            self.grid_world.get_mdp(),
            epsilon=config.VALUE_ITERATION_EPSILON
        )
        self.solver_generator = None

        # Control state
        self.running = True
        self.paused = True
        self.step_mode = False

        print("MDP Visualization")
        print("=" * 60)
        print("Grid World:")
        print(f"  Size: {self.grid_world.grid_size}x{self.grid_world.grid_size}")
        print(f"  Start: {self.grid_world.start_pos}")
        print(f"  Goal: {self.grid_world.goal_pos} (Reward: +1)")
        print(f"  Danger: {self.grid_world.danger_pos} (Reward: -1)")
        print(f"  Obstacles: {len(self.grid_world.obstacles)}")
        print(f"\nMDP Parameters:")
        print(f"  Discount (γ): {config.DISCOUNT}")
        print(f"  Noise: {config.NOISE}")
        print(f"  Living Reward: {config.LIVING_REWARD}")
        print(f"\nPress SPACE to start value iteration")
        print("=" * 60)

    def reset(self):
        """Reset environment and solver"""
        print("\nResetting environment...")
        self.grid_world = GridWorld(
            grid_size=config.GRID_SIZE,
            num_obstacles=config.NUM_OBSTACLES,
            noise=config.NOISE,
            discount=config.DISCOUNT,
            living_reward=config.LIVING_REWARD
        )
        self.visualizer = MDPVisualizer(self.grid_world)
        self.solver = ValueIteration(
            self.grid_world.get_mdp(),
            epsilon=config.VALUE_ITERATION_EPSILON
        )
        self.solver_generator = None
        self.paused = True
        print("Environment reset!")

    def start_solver(self):
        """Start solver iterations"""
        if self.solver_generator is None:
            self.solver_generator = self.solver.iterate()
            self.paused = False
            print("\nStarting value iteration...")

    def step_solver(self):
        """Execute one iteration"""
        if self.solver_generator:
            try:
                state = next(self.solver_generator)
                self.visualizer.update_state(state)

                if state.get('converged'):
                    print(f"\n✓ Converged after {state['iteration']} iterations!")
                    self.paused = True
            except StopIteration:
                self.paused = True
                print("\nSolver complete!")

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.solver_generator is None:
                        self.start_solver()
                    else:
                        self.paused = not self.paused
                        print("Paused" if self.paused else "Resumed")

                elif event.key == pygame.K_s:
                    if self.solver_generator is None:
                        self.start_solver()
                    self.step_mode = True

                elif event.key == pygame.K_r:
                    self.reset()

                elif event.key == pygame.K_v:
                    config.SHOW_VALUES = not config.SHOW_VALUES
                    print(f"Values: {'ON' if config.SHOW_VALUES else 'OFF'}")

                elif event.key == pygame.K_p:
                    config.SHOW_POLICY = not config.SHOW_POLICY
                    print(f"Policy: {'ON' if config.SHOW_POLICY else 'OFF'}")

                elif event.key == pygame.K_q:
                    config.SHOW_Q_VALUES = not config.SHOW_Q_VALUES
                    print(f"Q-values: {'ON' if config.SHOW_Q_VALUES else 'OFF'}")

    def update(self):
        """Update application"""
        if not self.paused and not self.solver.converged:
            self.step_solver()
            time.sleep(config.ITERATION_DELAY / config.ANIMATION_SPEED)

        elif self.step_mode:
            self.step_solver()
            self.step_mode = False

    def render(self):
        """Render application"""
        self.visualizer.render()

    def run(self):
        """Main loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()

        pygame.quit()
        sys.exit()


def main():
    """Entry point"""
    app = MDPApp()
    app.run()


if __name__ == '__main__':
    main()
