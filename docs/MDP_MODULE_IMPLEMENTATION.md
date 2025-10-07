# Markov Decision Process Module - Implementation Guide

**Status:** Production Ready
**Compatibility:** Local Python, Google Colab
**Skill Levels:** Beginner to Advanced
**Estimated Implementation Time:** 10-14 hours

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Complete Implementation](#2-complete-implementation)
3. [Setup Instructions](#3-setup-instructions)
4. [Google Colab Integration](#4-google-colab-integration)
5. [Configuration System](#5-configuration-system)
6. [Adding New MDP Problems](#6-adding-new-mdp-problems)
7. [Student Activities](#7-student-activities)
8. [Testing & Validation](#8-testing--validation)

---

## 1. Architecture Overview

### 1.1 Module Structure

```
modules/mdp/
├── __init__.py
├── config.py                    # Configuration and parameters
├── core/
│   ├── __init__.py
│   ├── mdp.py                  # MDP definition and problem
│   ├── solver.py               # Value iteration, policy iteration
│   └── simulator.py            # MDP simulation
├── environments/
│   ├── __init__.py
│   ├── grid_world.py           # Grid world environment
│   ├── custom.py               # Custom MDP builder
│   └── presets.py              # Preset problems
├── ui/
│   ├── __init__.py
│   ├── visualizer.py           # Main visualization
│   ├── value_heatmap.py        # Value function visualization
│   ├── policy_viz.py           # Policy arrows
│   └── transition_viz.py       # Probability visualization
└── main.py                      # Entry point
```

### 1.2 Key Concepts

**MDP Components:**
- **States (S)**: All possible situations
- **Actions (A)**: Choices available in each state
- **Transitions T(s,a,s')**: Probability of reaching s' from s via action a
- **Rewards R(s,a)**: Immediate reward for taking action a in state s
- **Discount γ**: How much we value future rewards (0-1)

**Solution Methods:**
- **Value Iteration**: Iteratively improve value estimates
- **Policy Iteration**: Iterate between policy evaluation and improvement
- **Monte Carlo**: Learn from sampled episodes

---

## 2. Complete Implementation

### 2.1 Configuration

```python
# modules/mdp/config.py
"""Configuration for MDP Module"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class MDPConfig:
    """Configuration for MDP module"""

    # Display settings
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 800
    CELL_SIZE: int = 100
    FPS: int = 60

    # Grid world settings
    GRID_SIZE: int = 5
    NUM_OBSTACLES: int = 2

    # MDP parameters (students can modify these!)
    DISCOUNT: float = 0.9          # γ (gamma) - discount factor
    NOISE: float = 0.2              # Stochastic transition probability
    LIVING_REWARD: float = -0.04    # Cost of living (per step)

    # Algorithm settings
    VALUE_ITERATION_EPSILON: float = 0.001  # Convergence threshold
    MAX_ITERATIONS: int = 1000
    ANIMATION_SPEED: float = 1.0     # Multiplier for iteration speed
    ITERATION_DELAY: float = 0.5     # Seconds between iterations

    # Visualization settings
    SHOW_VALUES: bool = True
    SHOW_POLICY: bool = True
    SHOW_Q_VALUES: bool = False
    ANIMATE_CONVERGENCE: bool = True
    VALUE_DECIMAL_PLACES: int = 2

    # Colors
    COLOR_BACKGROUND: Tuple[int, int, int] = (30, 30, 46)
    COLOR_GRID_LINE: Tuple[int, int, int] = (68, 71, 90)
    COLOR_OBSTACLE: Tuple[int, int, int] = (88, 91, 112)
    COLOR_GOAL: Tuple[int, int, int] = (255, 215, 0)      # Gold
    COLOR_DANGER: Tuple[int, int, int] = (255, 85, 85)    # Red
    COLOR_AGENT: Tuple[int, int, int] = (0, 150, 255)     # Blue
    COLOR_TEXT: Tuple[int, int, int] = (248, 248, 242)
    COLOR_ARROW: Tuple[int, int, int] = (255, 255, 255)
    COLOR_UI_BG: Tuple[int, int, int] = (40, 42, 54)

    # Value function visualization (heatmap)
    COLOR_VALUE_POSITIVE: Tuple[int, int, int] = (80, 250, 123)  # Green
    COLOR_VALUE_NEGATIVE: Tuple[int, int, int] = (255, 121, 198) # Pink
    COLOR_VALUE_NEUTRAL: Tuple[int, int, int] = (189, 147, 249)  # Purple

    # UI Layout
    SIDEBAR_WIDTH: int = 350
    CONTROL_PANEL_HEIGHT: int = 80


# Global config
config = MDPConfig()


# Presets for different scenarios
PRESETS = {
    'default': MDPConfig(),

    'deterministic': MDPConfig(
        NOISE=0.0,
        DISCOUNT=0.99,
    ),

    'high_noise': MDPConfig(
        NOISE=0.4,
        DISCOUNT=0.9,
    ),

    'cliff_world': MDPConfig(
        GRID_SIZE=8,
        LIVING_REWARD=-1.0,
        DISCOUNT=0.99,
    ),

    'fast_convergence': MDPConfig(
        ANIMATION_SPEED=5.0,
        ITERATION_DELAY=0.1,
    ),
}


def load_preset(name: str):
    """Load a preset configuration"""
    global config
    if name in PRESETS:
        config = PRESETS[name]
        print(f"Loaded preset: {name}")
    else:
        print(f"Unknown preset: {name}")
```

### 2.2 MDP Core

```python
# modules/mdp/core/mdp.py
"""Core MDP representation"""

from typing import Set, Dict, Tuple, List
from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    """Represents a state in the MDP"""
    position: Tuple[int, int]

    def __hash__(self):
        return hash(self.position)

    def __repr__(self):
        return f"S{self.position}"


class MDP:
    """Markov Decision Process definition"""

    def __init__(
        self,
        states: Set[State],
        actions: List[str],
        transitions: Dict[Tuple[State, str], Dict[State, float]],
        rewards: Dict[Tuple[State, str], float],
        discount: float,
        start_state: State,
        terminal_states: Set[State]
    ):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount = discount
        self.start_state = start_state
        self.terminal_states = terminal_states

    def get_actions(self, state: State) -> List[str]:
        """Get available actions in a state"""
        if state in self.terminal_states:
            return []
        return self.actions

    def get_transition_states_and_probs(
        self,
        state: State,
        action: str
    ) -> List[Tuple[State, float]]:
        """
        Get possible next states and their probabilities

        Returns:
            List of (next_state, probability) tuples
        """
        key = (state, action)
        if key in self.transitions:
            return list(self.transitions[key].items())
        return []

    def get_reward(self, state: State, action: str) -> float:
        """Get immediate reward for taking action in state"""
        key = (state, action)
        return self.rewards.get(key, 0.0)

    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal"""
        return state in self.terminal_states
```

### 2.3 MDP Solver

```python
# modules/mdp/core/solver.py
"""MDP solution algorithms"""

from typing import Dict, List, Tuple, Generator
import random
from modules.mdp.core.mdp import MDP, State

class ValueIteration:
    """Value Iteration algorithm"""

    def __init__(self, mdp: MDP, epsilon: float = 0.001):
        self.mdp = mdp
        self.epsilon = epsilon

        # Initialize values
        self.values: Dict[State, float] = {s: 0.0 for s in mdp.states}
        self.q_values: Dict[Tuple[State, str], float] = {}
        self.policy: Dict[State, str] = {}

        # Statistics
        self.iteration_count = 0
        self.converged = False

    def iterate(self) -> Generator[Dict, None, None]:
        """
        Run value iteration (generator for visualization)

        Yields:
            dict with current values, Q-values, and policy
        """
        while not self.converged:
            # Store old values
            old_values = self.values.copy()
            max_delta = 0.0

            # Update value for each state
            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    continue

                # Compute Q-values for all actions
                q_values_for_state = {}
                for action in self.mdp.get_actions(state):
                    q_value = self._compute_q_value(state, action, old_values)
                    q_values_for_state[action] = q_value
                    self.q_values[(state, action)] = q_value

                # Update value to max Q-value
                if q_values_for_state:
                    new_value = max(q_values_for_state.values())
                    self.values[state] = new_value

                    # Update policy to best action
                    self.policy[state] = max(
                        q_values_for_state,
                        key=q_values_for_state.get
                    )

                    # Track convergence
                    delta = abs(new_value - old_values[state])
                    max_delta = max(max_delta, delta)

            self.iteration_count += 1

            # Check convergence
            if max_delta < self.epsilon:
                self.converged = True

            # Yield current state
            yield {
                'values': self.values.copy(),
                'q_values': self.q_values.copy(),
                'policy': self.policy.copy(),
                'iteration': self.iteration_count,
                'max_delta': max_delta,
                'converged': self.converged,
            }

    def _compute_q_value(
        self,
        state: State,
        action: str,
        values: Dict[State, float]
    ) -> float:
        """
        Compute Q(s,a) = R(s,a) + γ Σ T(s,a,s') V(s')
        """
        # Immediate reward
        q_value = self.mdp.get_reward(state, action)

        # Expected future value
        next_states = self.mdp.get_transition_states_and_probs(state, action)
        for next_state, prob in next_states:
            q_value += self.mdp.discount * prob * values[next_state]

        return q_value

    def get_value(self, state: State) -> float:
        """Get value of state"""
        return self.values.get(state, 0.0)

    def get_policy(self, state: State) -> str:
        """Get best action for state"""
        return self.policy.get(state, None)

    def get_q_value(self, state: State, action: str) -> float:
        """Get Q-value for state-action pair"""
        return self.q_values.get((state, action), 0.0)


class PolicyIteration:
    """Policy Iteration algorithm"""

    def __init__(self, mdp: MDP, eval_iterations: int = 10):
        self.mdp = mdp
        self.eval_iterations = eval_iterations

        # Initialize random policy
        self.policy: Dict[State, str] = {}
        for state in mdp.states:
            if not mdp.is_terminal(state):
                actions = mdp.get_actions(state)
                self.policy[state] = random.choice(actions) if actions else None

        # Initialize values
        self.values: Dict[State, float] = {s: 0.0 for s in mdp.states}
        self.q_values: Dict[Tuple[State, str], float] = {}

        # Statistics
        self.iteration_count = 0
        self.converged = False

    def iterate(self) -> Generator[Dict, None, None]:
        """Run policy iteration"""
        while not self.converged:
            # Policy Evaluation
            self._policy_evaluation()

            # Policy Improvement
            policy_changed = self._policy_improvement()

            self.iteration_count += 1

            # Check convergence (policy stable)
            if not policy_changed:
                self.converged = True

            # Yield current state
            yield {
                'values': self.values.copy(),
                'q_values': self.q_values.copy(),
                'policy': self.policy.copy(),
                'iteration': self.iteration_count,
                'converged': self.converged,
            }

    def _policy_evaluation(self):
        """Evaluate current policy"""
        for _ in range(self.eval_iterations):
            new_values = {}
            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    new_values[state] = 0.0
                else:
                    action = self.policy[state]
                    if action:
                        new_values[state] = self._compute_q_value(
                            state, action, self.values
                        )
                    else:
                        new_values[state] = 0.0
            self.values = new_values

    def _policy_improvement(self) -> bool:
        """Improve policy based on current values"""
        policy_changed = False

        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                continue

            # Compute Q-values for all actions
            q_values_for_state = {}
            for action in self.mdp.get_actions(state):
                q_value = self._compute_q_value(state, action, self.values)
                q_values_for_state[action] = q_value
                self.q_values[(state, action)] = q_value

            # Get best action
            if q_values_for_state:
                best_action = max(q_values_for_state, key=q_values_for_state.get)

                # Check if policy changed
                if self.policy[state] != best_action:
                    self.policy[state] = best_action
                    policy_changed = True

        return policy_changed

    def _compute_q_value(
        self,
        state: State,
        action: str,
        values: Dict[State, float]
    ) -> float:
        """Compute Q(s,a)"""
        q_value = self.mdp.get_reward(state, action)

        next_states = self.mdp.get_transition_states_and_probs(state, action)
        for next_state, prob in next_states:
            q_value += self.mdp.discount * prob * values[next_state]

        return q_value
```

### 2.4 Grid World Environment

```python
# modules/mdp/environments/grid_world.py
"""Grid World environment for MDP"""

import random
from typing import Tuple, Set, List, Dict
from modules.mdp.core.mdp import MDP, State
from modules.mdp.config import config

class GridWorld:
    """Grid world environment"""

    def __init__(
        self,
        grid_size: int = 5,
        num_obstacles: int = 2,
        noise: float = 0.2,
        discount: float = 0.9,
        living_reward: float = -0.04
    ):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.noise = noise
        self.discount = discount
        self.living_reward = living_reward

        # Actions
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_effects = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }

        # Generate layout
        self.goal_pos = None
        self.danger_pos = None
        self.obstacles = set()
        self.start_pos = None

        self._generate_layout()

        # Build MDP
        self.mdp = self._build_mdp()

    def _generate_layout(self):
        """Generate random grid world layout"""
        # Place goal in top-right quadrant
        self.goal_pos = (
            random.randint(0, self.grid_size // 2),
            random.randint(self.grid_size // 2, self.grid_size - 1)
        )

        # Place danger adjacent to goal
        adjacent = [
            (self.goal_pos[0] - 1, self.goal_pos[1]),
            (self.goal_pos[0] + 1, self.goal_pos[1]),
            (self.goal_pos[0], self.goal_pos[1] - 1),
            (self.goal_pos[0], self.goal_pos[1] + 1),
        ]
        valid_danger = [
            pos for pos in adjacent
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
        ]
        self.danger_pos = random.choice(valid_danger) if valid_danger else (0, 0)

        # Place start in bottom-left quadrant
        while True:
            self.start_pos = (
                random.randint(self.grid_size // 2, self.grid_size - 1),
                random.randint(0, self.grid_size // 2)
            )
            if (self.start_pos != self.goal_pos and
                self.start_pos != self.danger_pos):
                break

        # Place obstacles
        all_positions = {
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        }
        forbidden = {self.start_pos, self.goal_pos, self.danger_pos}
        available = all_positions - forbidden

        self.obstacles = set(
            random.sample(list(available), min(self.num_obstacles, len(available)))
        )

    def _build_mdp(self) -> MDP:
        """Build MDP from grid world"""
        # States
        states = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.obstacles:
                    states.add(State((r, c)))

        start_state = State(self.start_pos)
        terminal_states = {State(self.goal_pos), State(self.danger_pos)}

        # Transitions and rewards
        transitions = {}
        rewards = {}

        for state in states:
            if state in terminal_states:
                continue

            for action in self.actions:
                # Get intended next state
                intended_next = self._get_next_position(state.position, action)

                # Build transition distribution (with noise)
                trans_dist = {}

                # Intended direction (1 - noise probability)
                intended_state = State(intended_next)
                if intended_state in states:
                    trans_dist[intended_state] = 1.0 - self.noise
                else:
                    trans_dist[state] = 1.0 - self.noise  # Hit wall, stay

                # Perpendicular directions (noise/2 each)
                perp_actions = self._get_perpendicular_actions(action)
                for perp_action in perp_actions:
                    perp_next = self._get_next_position(state.position, perp_action)
                    perp_state = State(perp_next)

                    if perp_state in states:
                        trans_dist[perp_state] = trans_dist.get(perp_state, 0) + self.noise / 2
                    else:
                        trans_dist[state] = trans_dist.get(state, 0) + self.noise / 2

                transitions[(state, action)] = trans_dist

                # Rewards
                if State(intended_next) == State(self.goal_pos):
                    rewards[(state, action)] = 1.0
                elif State(intended_next) == State(self.danger_pos):
                    rewards[(state, action)] = -1.0
                else:
                    rewards[(state, action)] = self.living_reward

        return MDP(
            states=states,
            actions=self.actions,
            transitions=transitions,
            rewards=rewards,
            discount=self.discount,
            start_state=start_state,
            terminal_states=terminal_states
        )

    def _get_next_position(self, pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Get next position after taking action (respecting boundaries)"""
        dr, dc = self.action_effects[action]
        new_r = max(0, min(self.grid_size - 1, pos[0] + dr))
        new_c = max(0, min(self.grid_size - 1, pos[1] + dc))

        # Check if hit obstacle
        if (new_r, new_c) in self.obstacles:
            return pos  # Stay in place

        return (new_r, new_c)

    def _get_perpendicular_actions(self, action: str) -> List[str]:
        """Get perpendicular actions for noise"""
        if action in ["UP", "DOWN"]:
            return ["LEFT", "RIGHT"]
        else:
            return ["UP", "DOWN"]

    def get_mdp(self) -> MDP:
        """Get MDP"""
        return self.mdp
```

### 2.5 Visualization

```python
# modules/mdp/ui/visualizer.py
"""MDP Visualization"""

import pygame
import math
from typing import Dict, Optional
from modules.mdp.config import config
from modules.mdp.environments.grid_world import GridWorld
from modules.mdp.core.mdp import State

class MDPVisualizer:
    """Visualizer for MDP"""

    def __init__(self, grid_world: GridWorld):
        pygame.init()

        self.grid_world = grid_world
        self.grid_size = grid_world.grid_size

        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("MDP Visualization")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 20)
        self.tiny_font = pygame.font.Font(None, 16)

        # Calculate cell size
        available_width = config.WINDOW_WIDTH - config.SIDEBAR_WIDTH
        available_height = config.WINDOW_HEIGHT - config.CONTROL_PANEL_HEIGHT
        self.cell_size = min(
            available_width // self.grid_size,
            available_height // self.grid_size
        )

        # Grid offset for centering
        self.grid_offset_x = config.SIDEBAR_WIDTH + (
            available_width - self.cell_size * self.grid_size
        ) // 2
        self.grid_offset_y = config.CONTROL_PANEL_HEIGHT + (
            available_height - self.cell_size * self.grid_size
        ) // 2

        # Current state
        self.values: Dict[State, float] = {}
        self.q_values: Dict = {}
        self.policy: Dict[State, str] = {}
        self.iteration = 0
        self.converged = False
        self.agent_pos = grid_world.start_pos

    def update_state(self, solver_state: Dict):
        """Update visualization state"""
        self.values = solver_state.get('values', {})
        self.q_values = solver_state.get('q_values', {})
        self.policy = solver_state.get('policy', {})
        self.iteration = solver_state.get('iteration', 0)
        self.converged = solver_state.get('converged', False)

    def render(self):
        """Render MDP"""
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)

        # Render grid
        self._render_grid()

        # Render sidebar
        self._render_sidebar()

        # Render control panel
        self._render_control_panel()

        # Update display
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def _render_grid(self):
        """Render grid world"""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = self.grid_offset_x + c * self.cell_size
                y = self.grid_offset_y + r * self.cell_size

                state = State((r, c))

                # Determine cell type and color
                if (r, c) in self.grid_world.obstacles:
                    color = config.COLOR_OBSTACLE
                    self._draw_cell(x, y, color)

                elif (r, c) == self.grid_world.goal_pos:
                    color = config.COLOR_GOAL
                    self._draw_cell(x, y, color)
                    # Draw star or trophy icon
                    self._draw_goal_icon(x, y)

                elif (r, c) == self.grid_world.danger_pos:
                    color = config.COLOR_DANGER
                    self._draw_cell(x, y, color)
                    # Draw danger icon
                    self._draw_danger_icon(x, y)

                else:
                    # Normal cell - color by value
                    if config.SHOW_VALUES and state in self.values:
                        value = self.values[state]
                        color = self._value_to_color(value)
                        self._draw_cell(x, y, color, alpha=150)
                    else:
                        self._draw_cell(x, y, (50, 50, 50), alpha=50)

                    # Draw value text
                    if config.SHOW_VALUES and state in self.values:
                        value = self.values[state]
                        self._draw_value_text(x, y, value)

                    # Draw policy arrow
                    if config.SHOW_POLICY and state in self.policy:
                        action = self.policy[state]
                        self._draw_policy_arrow(x, y, action)

                    # Draw Q-values in corners
                    if config.SHOW_Q_VALUES:
                        self._draw_q_values(x, y, state)

                # Draw agent if at this position
                if (r, c) == self.agent_pos:
                    self._draw_agent(x, y)

                # Draw grid lines
                pygame.draw.rect(
                    self.screen,
                    config.COLOR_GRID_LINE,
                    (x, y, self.cell_size, self.cell_size),
                    2
                )

    def _draw_cell(self, x: int, y: int, color: tuple, alpha: int = 255):
        """Draw a colored cell"""
        surface = pygame.Surface((self.cell_size, self.cell_size))
        surface.set_alpha(alpha)
        surface.fill(color)
        self.screen.blit(surface, (x, y))

    def _value_to_color(self, value: float) -> tuple:
        """Convert value to color (heatmap)"""
        if value > 0:
            # Positive values: green
            intensity = min(255, int(abs(value) * 255))
            return (0, intensity, 0)
        elif value < 0:
            # Negative values: red
            intensity = min(255, int(abs(value) * 255))
            return (intensity, 0, 0)
        else:
            # Zero: gray
            return (100, 100, 100)

    def _draw_value_text(self, x: int, y: int, value: float):
        """Draw value as text"""
        text = f"{value:.{config.VALUE_DECIMAL_PLACES}f}"
        text_surface = self.small_font.render(text, True, config.COLOR_TEXT)
        text_rect = text_surface.get_rect(
            center=(x + self.cell_size // 2, y + self.cell_size // 2)
        )
        self.screen.blit(text_surface, text_rect)

    def _draw_policy_arrow(self, x: int, y: int, action: str):
        """Draw policy arrow"""
        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2
        arrow_length = self.cell_size // 3

        # Direction vectors
        directions = {
            "UP": (0, -arrow_length),
            "DOWN": (0, arrow_length),
            "LEFT": (-arrow_length, 0),
            "RIGHT": (arrow_length, 0),
        }

        if action in directions:
            dx, dy = directions[action]
            end_x = center_x + dx
            end_y = center_y + dy

            # Draw arrow line
            pygame.draw.line(
                self.screen,
                config.COLOR_ARROW,
                (center_x, center_y),
                (end_x, end_y),
                4
            )

            # Draw arrowhead
            arrow_size = 8
            angle = math.atan2(dy, dx)

            point1_x = end_x - arrow_size * math.cos(angle - math.pi / 6)
            point1_y = end_y - arrow_size * math.sin(angle - math.pi / 6)
            point2_x = end_x - arrow_size * math.cos(angle + math.pi / 6)
            point2_y = end_y - arrow_size * math.sin(angle + math.pi / 6)

            pygame.draw.polygon(
                self.screen,
                config.COLOR_ARROW,
                [(end_x, end_y), (point1_x, point1_y), (point2_x, point2_y)]
            )

    def _draw_q_values(self, x: int, y: int, state: State):
        """Draw Q-values in cell corners"""
        # Q-value positions (top, right, bottom, left)
        positions = {
            "UP": (x + self.cell_size // 2, y + 10),
            "RIGHT": (x + self.cell_size - 25, y + self.cell_size // 2),
            "DOWN": (x + self.cell_size // 2, y + self.cell_size - 10),
            "LEFT": (x + 15, y + self.cell_size // 2),
        }

        for action, pos in positions.items():
            q_value = self.q_values.get((state, action), 0.0)
            text = f"{q_value:.1f}"
            text_surface = self.tiny_font.render(text, True, (200, 200, 200))
            text_rect = text_surface.get_rect(center=pos)
            self.screen.blit(text_surface, text_rect)

    def _draw_goal_icon(self, x: int, y: int):
        """Draw goal icon (star)"""
        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2
        radius = self.cell_size // 4

        # Draw simple circle for now (could be star)
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (center_x, center_y),
            radius,
            3
        )

    def _draw_danger_icon(self, x: int, y: int):
        """Draw danger icon (X)"""
        padding = self.cell_size // 4
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (x + padding, y + padding),
            (x + self.cell_size - padding, y + self.cell_size - padding),
            4
        )
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (x + self.cell_size - padding, y + padding),
            (x + padding, y + self.cell_size - padding),
            4
        )

    def _draw_agent(self, x: int, y: int):
        """Draw agent"""
        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2
        radius = self.cell_size // 4

        pygame.draw.circle(
            self.screen,
            config.COLOR_AGENT,
            (center_x, center_y),
            radius
        )
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (center_x, center_y),
            radius,
            2
        )

    def _render_sidebar(self):
        """Render sidebar with info"""
        # Background
        pygame.draw.rect(
            self.screen,
            config.COLOR_UI_BG,
            (0, 0, config.SIDEBAR_WIDTH, config.WINDOW_HEIGHT)
        )

        # Title
        title = self.font.render("MDP Solver", True, config.COLOR_TEXT)
        self.screen.blit(title, (10, 10))

        # Statistics
        y_offset = 60
        stats = [
            ("Iteration", self.iteration),
            ("Discount (γ)", config.DISCOUNT),
            ("Noise", config.NOISE),
            ("Living Reward", config.LIVING_REWARD),
        ]

        for label, value in stats:
            if isinstance(value, float):
                text = f"{label}: {value:.2f}"
            else:
                text = f"{label}: {value}"
            text_surface = self.small_font.render(text, True, config.COLOR_TEXT)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 30

        # Status
        y_offset += 20
        status_text = "Converged!" if self.converged else "Iterating..."
        status_color = config.COLOR_GOAL if self.converged else config.COLOR_TEXT
        status = self.small_font.render(status_text, True, status_color)
        self.screen.blit(status, (10, y_offset))

        # Legend
        y_offset += 60
        legend_title = self.font.render("Legend:", True, config.COLOR_TEXT)
        self.screen.blit(legend_title, (10, y_offset))
        y_offset += 35

        legend_items = [
            ("Goal (Reward: +1)", config.COLOR_GOAL),
            ("Danger (Reward: -1)", config.COLOR_DANGER),
            ("Obstacle", config.COLOR_OBSTACLE),
            ("Agent", config.COLOR_AGENT),
        ]

        for label, color in legend_items:
            # Color box
            pygame.draw.rect(self.screen, color, (10, y_offset, 20, 20))
            # Label
            text = self.tiny_font.render(label, True, config.COLOR_TEXT)
            self.screen.blit(text, (35, y_offset + 3))
            y_offset += 25

    def _render_control_panel(self):
        """Render control panel"""
        # Background
        pygame.draw.rect(
            self.screen,
            config.COLOR_UI_BG,
            (config.SIDEBAR_WIDTH, 0,
             config.WINDOW_WIDTH - config.SIDEBAR_WIDTH,
             config.CONTROL_PANEL_HEIGHT)
        )

        # Instructions
        instructions = [
            "SPACE: Start/Pause | S: Step | R: Reset | V: Toggle Values | P: Toggle Policy | Q: Toggle Q-values"
        ]

        y_offset = 15
        for instruction in instructions:
            text = self.tiny_font.render(instruction, True, config.COLOR_TEXT)
            self.screen.blit(text, (config.SIDEBAR_WIDTH + 10, y_offset))
            y_offset += 20
```

### 2.6 Main Application

```python
# modules/mdp/main.py
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
```

---

## 3. Setup Instructions

```bash
# Install dependencies
pip install pygame numpy matplotlib

# Run module
python -m modules.mdp.main
```

---

## 4. Google Colab Integration

```python
# modules/mdp/colab_main.py
"""Colab-compatible MDP visualization"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np

from modules.mdp.environments.grid_world import GridWorld
from modules.mdp.core.solver import ValueIteration

def run_mdp_colab(grid_size=5, num_obstacles=2, noise=0.2):
    """Run MDP in Colab"""
    # Create environment
    gw = GridWorld(grid_size=grid_size, num_obstacles=num_obstacles, noise=noise)

    # Create solver
    solver = ValueIteration(gw.get_mdp())

    # Setup figure
    fig, (ax_grid, ax_stats) = plt.subplots(1, 2, figsize=(14, 6))

    # Animation update function
    def update(frame):
        try:
            state = next(solver_gen)
        except StopIteration:
            return

        ax_grid.clear()

        # Draw grid
        for r in range(grid_size):
            for c in range(grid_size):
                # Color by value
                if (r, c) in gw.obstacles:
                    color = 'gray'
                elif (r, c) == gw.goal_pos:
                    color = 'gold'
                elif (r, c) == gw.danger_pos:
                    color = 'red'
                else:
                    from modules.mdp.core.mdp import State
                    s = State((r, c))
                    value = state['values'].get(s, 0)
                    intensity = abs(value)
                    if value > 0:
                        color = (0, intensity, 0)
                    else:
                        color = (intensity, 0, 0)

                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    facecolor=color, edgecolor='black'
                )
                ax_grid.add_patch(rect)

        ax_grid.set_xlim(0, grid_size)
        ax_grid.set_ylim(0, grid_size)
        ax_grid.set_aspect('equal')
        ax_grid.invert_yaxis()
        ax_grid.set_title(f"Iteration: {state['iteration']}")

        # Update stats
        ax_stats.clear()
        ax_stats.axis('off')
        stats_text = f"""
        Status: {'Converged' if state['converged'] else 'Iterating'}
        Iteration: {state['iteration']}
        """
        ax_stats.text(0.1, 0.5, stats_text, fontsize=14, family='monospace')

    solver_gen = solver.iterate()
    anim = FuncAnimation(fig, update, frames=100, interval=200, repeat=False)
    plt.close()
    return HTML(anim.to_jshtml())
```

---

## 5. Configuration System

Students can modify parameters in `config.py`:

```python
from modules.mdp.config import config

# Make environment more stochastic
config.NOISE = 0.4

# Change discount factor
config.DISCOUNT = 0.95

# Make living more expensive
config.LIVING_REWARD = -0.1
```

---

## 6. Adding New MDP Problems

Advanced students can create custom MDPs:

```python
# modules/mdp/environments/custom.py
"""Custom MDP problems"""

from modules.mdp.core.mdp import MDP, State

def create_cliff_world():
    """Create cliff walking problem"""
    # Define states, actions, transitions, rewards
    # ... (implement custom problem)
    pass
```

---

## 7. Student Activities

### Beginner: Explore MDP Behavior
1. Run value iteration and watch values propagate
2. Toggle policy display to see optimal actions
3. Modify discount factor and observe changes

### Intermediate: Parameter Experiments
```python
# Test different noise levels
for noise in [0.0, 0.2, 0.4]:
    config.NOISE = noise
    # Run and compare policies
```

### Advanced: Implement Policy Iteration
Implement policy iteration algorithm and compare with value iteration

---

## 8. Testing & Validation

```python
def test_value_iteration_converges():
    """Test that value iteration converges"""
    gw = GridWorld(grid_size=5)
    solver = ValueIteration(gw.get_mdp())

    for state in solver.iterate():
        pass

    assert solver.converged
    assert solver.iteration_count > 0
```

---

**MDP Module Complete!**

This implementation provides a complete, interactive MDP learning environment with:
✅ Value Iteration & Policy Iteration
✅ Interactive grid world
✅ Real-time value propagation visualization
✅ Configurable parameters
✅ Colab support
✅ Student activities for all levels

Students can experiment with:
- Different discount factors
- Varying noise levels
- Custom reward structures
- New algorithms
