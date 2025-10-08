"""Main application for MDP module with enhanced features"""

import pygame
import sys
import time
import random
from modules.mdp.config import config
from modules.mdp.environments.grid_world import GridWorld
from modules.mdp.core.solver import ValueIteration
from modules.mdp.core.mdp import State
from modules.mdp.ui.visualizer import MDPVisualizer


class MDPApp:
    """Enhanced MDP application with policy demo and learning animation"""

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

        # Create solver and run value iteration to get policy
        mdp = self.grid_world.get_mdp()
        self.solver = ValueIteration(mdp, epsilon=config.VALUE_ITERATION_EPSILON)

        # Run value iteration to completion and store all iterations
        self.iterations = []
        for state in self.solver.iterate():
            self.iterations.append(state)

        # Extract final values and policy
        if self.iterations:
            final_state = self.iterations[-1]
            self.visualizer.update_state(final_state)

        # Control state
        self.running = True
        self.show_learning_process = False
        self.policy_demo_mode = False
        self.current_iteration = 0
        self.walker_pos = self.grid_world.start_pos

        # Terminal state message
        self.terminal_state_message = ""

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
        print(f"\nValue iteration converged in {self.solver.iteration_count} iterations")
        print("\nControls:")
        print("  V: Toggle values | P: Toggle policy | Q: Toggle Q-values")
        print("  L: Show learning process | D: Policy demo | R: Reset")
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

        # Rerun value iteration
        mdp = self.grid_world.get_mdp()
        self.solver = ValueIteration(mdp, epsilon=config.VALUE_ITERATION_EPSILON)

        self.iterations = []
        for state in self.solver.iterate():
            self.iterations.append(state)

        if self.iterations:
            final_state = self.iterations[-1]
            self.visualizer.update_state(final_state)

        self.walker_pos = self.grid_world.start_pos
        self.current_iteration = 0
        self.show_learning_process = False
        self.policy_demo_mode = False
        self.terminal_state_message = ""

        print(f"Environment reset! Converged in {self.solver.iteration_count} iterations")

    def toggle_policy_demo(self):
        """Toggle policy demo mode"""
        self.policy_demo_mode = not self.policy_demo_mode

        if self.policy_demo_mode:
            # Start walker at random non-terminal position
            valid_starts = [
                s.position for s in self.grid_world.mdp.states
                if s not in self.grid_world.mdp.terminal_states and s.position not in self.grid_world.obstacles
            ]
            if valid_starts:
                self.walker_pos = random.choice(valid_starts)
            else:
                self.walker_pos = self.grid_world.start_pos

            self.terminal_state_message = ""
            print("\n✓ Policy demo started - watch the agent follow optimal policy!")
        else:
            self.walker_pos = self.grid_world.start_pos
            self.terminal_state_message = ""
            print("Policy demo stopped")

    def toggle_learning_animation(self):
        """Toggle learning process animation"""
        self.show_learning_process = not self.show_learning_process
        self.current_iteration = 0

        if self.show_learning_process:
            print("\n✓ Showing learning process - watch values converge!")
        else:
            # Show final state
            if self.iterations:
                self.visualizer.update_state(self.iterations[-1])
            print("Learning animation stopped")

    def update_policy_demo(self):
        """Update policy demo - move walker"""
        current_state = State(self.walker_pos)

        # Check if at terminal state
        if current_state in self.grid_world.mdp.terminal_states:
            if self.walker_pos == self.grid_world.goal_pos:
                self.terminal_state_message = "Reached the Goal! Episode Ended."
            elif self.walker_pos == self.grid_world.danger_pos:
                self.terminal_state_message = "Fell into Danger! Episode Ended."

            pygame.time.wait(1000)

            # Reset to random position
            valid_starts = [
                s.position for s in self.grid_world.mdp.states
                if s not in self.grid_world.mdp.terminal_states and s.position not in self.grid_world.obstacles
            ]
            if valid_starts:
                self.walker_pos = random.choice(valid_starts)
            self.terminal_state_message = ""
        else:
            # Follow policy
            self.terminal_state_message = ""

            if current_state in self.solver.policy:
                action = self.solver.policy[current_state]

                # Get next state based on stochastic transition
                transitions = self.grid_world.mdp.get_transition_states_and_probs(current_state, action)
                if transitions:
                    next_states = [s.position for s, p in transitions]
                    probabilities = [p for s, p in transitions]
                    self.walker_pos = random.choices(next_states, weights=probabilities, k=1)[0]

            pygame.time.wait(500)  # Delay between steps

    def update_learning_animation(self):
        """Update learning process animation"""
        if self.current_iteration < len(self.iterations):
            self.visualizer.update_state(self.iterations[self.current_iteration])
            self.current_iteration += 1

            if self.current_iteration == len(self.iterations):
                self.show_learning_process = False
                print("\n✓ Learning animation complete!")

            pygame.time.wait(200)  # Delay between iterations

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    config.SHOW_VALUES = not config.SHOW_VALUES
                    print(f"Values: {'ON' if config.SHOW_VALUES else 'OFF'}")

                elif event.key == pygame.K_p or event.key == pygame.K_SPACE:
                    config.SHOW_POLICY = not config.SHOW_POLICY
                    print(f"Policy: {'ON' if config.SHOW_POLICY else 'OFF'}")

                elif event.key == pygame.K_q:
                    config.SHOW_Q_VALUES = not config.SHOW_Q_VALUES
                    print(f"Q-values: {'ON' if config.SHOW_Q_VALUES else 'OFF'}")

                elif event.key == pygame.K_l:
                    self.toggle_learning_animation()

                elif event.key == pygame.K_d:
                    self.toggle_policy_demo()

                elif event.key == pygame.K_r:
                    self.reset()

                # Manual control with arrow keys (when not in demo mode)
                elif not self.policy_demo_mode:
                    action_map = {
                        pygame.K_UP: "UP",
                        pygame.K_DOWN: "DOWN",
                        pygame.K_LEFT: "LEFT",
                        pygame.K_RIGHT: "RIGHT"
                    }

                    if event.key in action_map:
                        action = action_map[event.key]
                        current_state = State(self.walker_pos)

                        # Get next state
                        transitions = self.grid_world.mdp.get_transition_states_and_probs(current_state, action)
                        if transitions:
                            next_states = [s.position for s, p in transitions]
                            probabilities = [p for s, p in transitions]
                            self.walker_pos = random.choices(next_states, weights=probabilities, k=1)[0]

                            print(f"Moved {action} to {self.walker_pos}")

                            # Check terminal
                            if self.walker_pos == self.grid_world.goal_pos:
                                print("Reached goal!")
                                self.walker_pos = self.grid_world.start_pos
                            elif self.walker_pos == self.grid_world.danger_pos:
                                print("Hit danger!")
                                self.walker_pos = self.grid_world.start_pos

    def update(self):
        """Update application"""
        if self.show_learning_process:
            self.update_learning_animation()

        if self.policy_demo_mode:
            self.update_policy_demo()

    def render(self):
        """Render application"""
        # Render main visualization
        self.visualizer.render(walker_pos=self.walker_pos)

        # Draw status message if any
        if self.terminal_state_message:
            text = self.visualizer.font.render(
                self.terminal_state_message,
                True,
                (255, 0, 0)
            )
            # Draw at bottom center
            x = (config.WINDOW_WIDTH - text.get_width()) // 2
            y = config.WINDOW_HEIGHT - 60
            self.screen.blit(text, (x, y))

        # Draw mode indicator
        mode = "Policy Demo" if self.policy_demo_mode else "Manual Control"
        mode_text = self.visualizer.small_font.render(
            f"Mode: {mode}",
            True,
            config.COLOR_TEXT
        )
        self.screen.blit(mode_text, (config.SIDEBAR_WIDTH + 10, config.WINDOW_HEIGHT - 30))

        # Access screen from visualizer
        self.screen = self.visualizer.screen

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
