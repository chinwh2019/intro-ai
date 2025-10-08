"""
MDP Module - Web Version
Runs in browser via Pygbag/WebAssembly

Controls:
  V: Toggle values | P: Toggle policy | Q: Toggle Q-values
  L: Learning animation | SPACE: Pause | S: Step
  D: Policy demo | M: Manual learning | R: Reset
  Arrow Keys: Move agent
"""

# /// script
# [pygbag]
# autorun = true
# ///

import asyncio
import pygame
import sys
import random
from typing import Optional

# Import from local modules (no 'modules.mdp' prefix for web)
import config
from core.mdp import State
from core.solver import ValueIteration
from core.learner import MDPLearner
from environments.grid_world import GridWorld
from ui.visualizer import MDPVisualizer


class MDPApp:
    """Web-compatible MDP application with async/await"""

    def __init__(self):
        # Create environment
        self.grid_world = GridWorld(
            grid_size=config.config.GRID_SIZE,
            num_obstacles=config.config.NUM_OBSTACLES,
            noise=config.config.NOISE,
            discount=config.config.DISCOUNT,
            living_reward=config.config.LIVING_REWARD
        )

        # Create visualizer with parameter change callback
        self.visualizer = MDPVisualizer(self.grid_world, on_parameter_change=self.on_parameter_change)

        # Create solver and run value iteration
        mdp = self.grid_world.get_mdp()
        self.solver = ValueIteration(mdp, epsilon=config.config.VALUE_ITERATION_EPSILON)

        # Run value iteration and store all iterations
        self.iterations = []
        for state in self.solver.iterate():
            self.iterations.append(state)

        # Show initial state (zeros except terminals)
        initial_state = {
            'values': {s: self.solver.values[s] if self.solver.mdp.is_terminal(s) else 0.0
                      for s in self.solver.mdp.states},
            'q_values': {(s, a): self.solver.values[s] if self.solver.mdp.is_terminal(s) else 0.0
                        for s in self.solver.mdp.states for a in self.solver.mdp.actions},
            'policy': {},
            'iteration': 0,
            'converged': False,
        }
        self.visualizer.update_state(initial_state)

        # Control state
        self.running = True
        self.show_learning_process = False
        self.learning_paused = False
        self.step_mode = False
        self.policy_demo_mode = False
        self.manual_learning_mode = False
        self.current_iteration = 0
        self.walker_pos = self.grid_world.start_pos

        # Manual learning
        self.learned_V = None
        self.learned_Q = None
        self.episode_count = 0

        # Terminal state message
        self.terminal_state_message = ""

        # Screen reference
        self.screen = self.visualizer.screen

        # Time tracking for async updates
        self.last_policy_demo_time = 0
        self.last_learning_time = 0

        print("MDP Visualization - Web Version")
        print("=" * 60)
        print("Converged in", self.solver.iteration_count, "iterations")
        print("Press V/P/Q to toggle displays")
        print("Press L for learning, D for demo, M for manual learning")

    def on_parameter_change(self, params: dict):
        """Handle parameter change from sliders"""
        try:
            print("\n🔄 Applying new parameters...")
            print(f"  Discount: {params['discount']:.2f}")
            print(f"  Noise: {params['noise']:.2f}")
            print(f"  Living reward: {params['living_reward']:.3f}")

            # Update config
            config.config.DISCOUNT = params['discount']
            config.config.NOISE = params['noise']
            config.config.LIVING_REWARD = params['living_reward']

            # Re-solve MDP
            self._resolve_mdp()

            print("✓ Parameters applied successfully!")
        except Exception as e:
            print(f"❌ Error applying parameters: {e}")
            import traceback
            traceback.print_exc()

    def _resolve_mdp(self):
        """Re-solve MDP with new parameters"""
        # Keep same layout, rebuild with new parameters
        mdp = GridWorld(
            grid_size=self.grid_world.grid_size,
            num_obstacles=len(self.grid_world.obstacles),
            noise=config.config.NOISE,
            discount=config.config.DISCOUNT,
            living_reward=config.config.LIVING_REWARD
        )

        mdp.goal_pos = self.grid_world.goal_pos
        mdp.danger_pos = self.grid_world.danger_pos
        mdp.start_pos = self.grid_world.start_pos
        mdp.obstacles = self.grid_world.obstacles.copy()
        mdp.mdp = mdp._build_mdp()

        self.grid_world = mdp

        # Re-solve
        self.solver = ValueIteration(mdp.get_mdp(), epsilon=config.config.VALUE_ITERATION_EPSILON)

        self.iterations = []
        for state in self.solver.iterate():
            self.iterations.append(state)

        initial_state = {
            'values': {s: self.solver.values[s] if self.solver.mdp.is_terminal(s) else 0.0
                      for s in self.solver.mdp.states},
            'q_values': {},
            'policy': {},
            'iteration': 0,
            'converged': False,
        }
        self.visualizer.update_state(initial_state)

        # Update sliders
        self.visualizer.parameter_panel.set_parameters(
            discount=config.config.DISCOUNT,
            noise=config.config.NOISE,
            living_reward=config.config.LIVING_REWARD
        )

    def reset(self):
        """Reset environment"""
        print("\nResetting...")
        self.grid_world = GridWorld(
            grid_size=config.config.GRID_SIZE,
            num_obstacles=config.config.NUM_OBSTACLES,
            noise=config.config.NOISE,
            discount=config.config.DISCOUNT,
            living_reward=config.config.LIVING_REWARD
        )
        self.visualizer = MDPVisualizer(self.grid_world)

        mdp = self.grid_world.get_mdp()
        self.solver = ValueIteration(mdp, epsilon=config.config.VALUE_ITERATION_EPSILON)

        self.iterations = []
        for state in self.solver.iterate():
            self.iterations.append(state)

        initial_state = {
            'values': {s: self.solver.values[s] if self.solver.mdp.is_terminal(s) else 0.0
                      for s in self.solver.mdp.states},
            'q_values': {(s, a): self.solver.values[s] if self.solver.mdp.is_terminal(s) else 0.0
                        for s in self.solver.mdp.states for a in self.solver.mdp.actions},
            'policy': {},
            'iteration': 0,
            'converged': False,
        }
        self.visualizer.update_state(initial_state)

        self.walker_pos = self.grid_world.start_pos
        self.current_iteration = 0
        self.show_learning_process = False
        self.learning_paused = False
        self.step_mode = False
        self.policy_demo_mode = False
        self.manual_learning_mode = False
        self.terminal_state_message = ""
        self.screen = self.visualizer.screen

    def toggle_policy_demo(self):
        """Toggle policy demo mode"""
        self.policy_demo_mode = not self.policy_demo_mode
        self.manual_learning_mode = False

        if self.policy_demo_mode:
            if self.iterations:
                self.visualizer.update_state(self.iterations[-1])

            valid_starts = [
                s.position for s in self.grid_world.mdp.states
                if s not in self.grid_world.mdp.terminal_states and s.position not in self.grid_world.obstacles
            ]
            self.walker_pos = random.choice(valid_starts) if valid_starts else self.grid_world.start_pos
            print("\n✓ Policy demo started")
        else:
            initial_state = {
                'values': {s: self.solver.values[s] if self.solver.mdp.is_terminal(s) else 0.0
                          for s in self.solver.mdp.states},
                'q_values': {},
                'policy': {},
                'iteration': 0,
                'converged': False,
            }
            self.visualizer.update_state(initial_state)
            self.walker_pos = self.grid_world.start_pos

    def toggle_learning_animation(self):
        """Toggle learning animation"""
        self.show_learning_process = not self.show_learning_process
        self.current_iteration = 0
        self.learning_paused = False

        if not self.show_learning_process:
            initial_state = {
                'values': {s: self.solver.values[s] if self.solver.mdp.is_terminal(s) else 0.0
                          for s in self.solver.mdp.states},
                'q_values': {},
                'policy': {},
                'iteration': 0,
                'converged': False,
            }
            self.visualizer.update_state(initial_state)

    def toggle_manual_learning(self):
        """Toggle manual learning"""
        self.manual_learning_mode = not self.manual_learning_mode
        self.policy_demo_mode = False

        if self.manual_learning_mode:
            self.reset_learning()
            if not config.config.SHOW_Q_VALUES:
                config.config.SHOW_Q_VALUES = True
            config.config.SHOW_POLICY = False
            print("\n✓ Manual Learning Mode")
        else:
            initial_state = {
                'values': {s: self.solver.values[s] if self.solver.mdp.is_terminal(s) else 0.0
                          for s in self.solver.mdp.states},
                'q_values': {},
                'policy': {},
                'iteration': 0,
                'converged': False,
            }
            self.visualizer.update_state(initial_state)
            self.walker_pos = self.grid_world.start_pos

    def reset_learning(self):
        """Reset learned values"""
        self.learned_V = {s: 0.0 for s in self.grid_world.mdp.states}
        self.learned_Q = {s: {a: 0.0 for a in self.grid_world.mdp.actions}
                         for s in self.grid_world.mdp.states}

        for state in self.grid_world.mdp.terminal_states:
            if state in self.grid_world.mdp.terminal_rewards:
                reward = self.grid_world.mdp.terminal_rewards[state]
                self.learned_V[state] = reward
                for action in self.grid_world.mdp.actions:
                    self.learned_Q[state][action] = reward

        self.walker_pos = self.grid_world.start_pos
        self.episode_count = 0
        self._update_learned_display()

    def _update_learned_display(self):
        """Update display with learned values"""
        if not self.manual_learning_mode or self.learned_Q is None:
            return

        q_values_display = {}
        for state, actions in self.learned_Q.items():
            for action, value in actions.items():
                q_values_display[(state, action)] = value

        display_state = {
            'values': self.learned_V.copy() if self.learned_V else {},
            'q_values': q_values_display,
            'policy': {},
            'iteration': self.episode_count,
            'converged': False,
        }
        self.visualizer.update_state(display_state)

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            # Pass event to parameter panel first
            self.visualizer.handle_parameter_event(event)

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.show_learning_process:
                        self.learning_paused = not self.learning_paused
                    else:
                        config.config.SHOW_POLICY = not config.config.SHOW_POLICY

                elif event.key == pygame.K_s:
                    if self.show_learning_process and self.learning_paused:
                        self.step_mode = True

                elif event.key == pygame.K_v:
                    config.config.SHOW_VALUES = not config.config.SHOW_VALUES

                elif event.key == pygame.K_p:
                    config.config.SHOW_POLICY = not config.config.SHOW_POLICY

                elif event.key == pygame.K_q:
                    config.config.SHOW_Q_VALUES = not config.config.SHOW_Q_VALUES

                elif event.key == pygame.K_l:
                    self.toggle_learning_animation()

                elif event.key == pygame.K_d:
                    self.toggle_policy_demo()

                elif event.key == pygame.K_m:
                    self.toggle_manual_learning()

                elif event.key == pygame.K_r:
                    self.reset()

                # Arrow keys
                elif not self.policy_demo_mode and not self.show_learning_process:
                    action_map = {
                        pygame.K_UP: "UP",
                        pygame.K_DOWN: "DOWN",
                        pygame.K_LEFT: "LEFT",
                        pygame.K_RIGHT: "RIGHT"
                    }

                    if event.key in action_map:
                        action = action_map[event.key]
                        current_state = State(self.walker_pos)
                        reward = self.grid_world.mdp.get_reward(current_state, action)

                        transitions = self.grid_world.mdp.get_transition_states_and_probs(current_state, action)
                        if transitions:
                            next_states = [s.position for s, p in transitions]
                            probabilities = [p for s, p in transitions]
                            next_pos = random.choices(next_states, weights=probabilities, k=1)[0]
                            next_state = State(next_pos)

                            if self.manual_learning_mode:
                                MDPLearner.update_values(
                                    current_state, action, next_state, reward,
                                    self.learned_V, self.learned_Q,
                                    config.config.DISCOUNT, config.config.LEARNING_RATE
                                )

                            self.walker_pos = next_pos

                            if self.walker_pos == self.grid_world.goal_pos:
                                if self.manual_learning_mode:
                                    self.episode_count += 1
                                    self._update_learned_display()
                                self.walker_pos = self.grid_world.start_pos
                            elif self.walker_pos == self.grid_world.danger_pos:
                                if self.manual_learning_mode:
                                    self.episode_count += 1
                                    self._update_learned_display()
                                self.walker_pos = self.grid_world.start_pos
                            elif self.manual_learning_mode:
                                self._update_learned_display()

    def update(self, current_time: float):
        """Update with time-based logic for web"""
        # Learning animation
        if self.show_learning_process and (not self.learning_paused or self.step_mode):
            if current_time - self.last_learning_time >= 0.2:  # 200ms delay
                if self.current_iteration < len(self.iterations):
                    self.visualizer.update_state(self.iterations[self.current_iteration])
                    self.current_iteration += 1

                    if self.current_iteration == len(self.iterations):
                        self.show_learning_process = False
                        self.learning_paused = False

                    self.step_mode = False
                    self.last_learning_time = current_time

        # Policy demo
        if self.policy_demo_mode:
            if current_time - self.last_policy_demo_time >= 0.5:  # 500ms delay
                current_state = State(self.walker_pos)

                if current_state in self.grid_world.mdp.terminal_states:
                    valid_starts = [
                        s.position for s in self.grid_world.mdp.states
                        if s not in self.grid_world.mdp.terminal_states and s.position not in self.grid_world.obstacles
                    ]
                    self.walker_pos = random.choice(valid_starts) if valid_starts else self.grid_world.start_pos
                    self.terminal_state_message = ""
                elif current_state in self.solver.policy:
                    action = self.solver.policy[current_state]
                    transitions = self.grid_world.mdp.get_transition_states_and_probs(current_state, action)
                    if transitions:
                        next_states = [s.position for s, p in transitions]
                        probabilities = [p for s, p in transitions]
                        self.walker_pos = random.choices(next_states, weights=probabilities, k=1)[0]

                self.last_policy_demo_time = current_time

    def render(self):
        """Render application"""
        self.visualizer.render(walker_pos=self.walker_pos)

        # Draw terminal message
        if self.terminal_state_message:
            text = self.visualizer.font.render(self.terminal_state_message, True, (255, 0, 0))
            x = (config.config.WINDOW_WIDTH - text.get_width()) // 2
            y = config.config.WINDOW_HEIGHT - 60
            self.screen.blit(text, (x, y))

        # Draw mode indicator
        if self.manual_learning_mode:
            mode = f"Manual Learning (Episode {self.episode_count})"
            mode_color = (80, 250, 123)
        elif self.policy_demo_mode:
            mode = "Policy Demo"
            mode_color = (139, 233, 253)
        elif self.show_learning_process:
            mode = f"Learning ({self.visualizer.iteration}/{len(self.iterations)})"
            mode_color = (255, 184, 108)
        else:
            mode = "Observation"
            mode_color = config.config.COLOR_TEXT

        mode_text = self.visualizer.small_font.render(f"Mode: {mode}", True, mode_color)
        self.screen.blit(mode_text, (config.config.SIDEBAR_WIDTH + 10, config.config.WINDOW_HEIGHT - 30))


async def main():
    """Async main loop for web compatibility"""
    print("Loading MDP Module...")

    app = MDPApp()
    clock = pygame.time.Clock()

    print("✓ Ready! Press L for learning, D for demo, M for manual")

    # Main game loop (MUST be async for web)
    while app.running:
        # Get current time
        current_time = pygame.time.get_ticks() / 1000.0

        # Process events and update
        app.handle_events()
        app.update(current_time)
        app.render()

        # CRITICAL: Yield control to browser event loop
        await asyncio.sleep(0)

        # Frame rate limiting
        clock.tick(config.config.FPS)

    pygame.quit()


# Entry point
if __name__ == '__main__':
    asyncio.run(main())
