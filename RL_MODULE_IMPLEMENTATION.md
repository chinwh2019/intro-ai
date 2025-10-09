# Reinforcement Learning Module - Implementation Guide

**Status:** Production Ready
**Compatibility:** Local Python, Google Colab
**Skill Levels:** Beginner to Advanced
**Estimated Implementation Time:** 12-16 hours
**Featured Environment:** Snake Game with Q-Learning

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Complete Implementation](#2-complete-implementation)
3. [Setup Instructions](#3-setup-instructions)
4. [Google Colab Integration](#4-google-colab-integration)
5. [Configuration System](#5-configuration-system)
6. [Adding New Environments](#6-adding-new-environments)
7. [Adding New Algorithms](#7-adding-new-algorithms)
8. [Student Activities](#8-student-activities)
9. [Testing & Validation](#9-testing--validation)
10. [Advanced Topics](#10-advanced-topics)

---

## 1. Architecture Overview

### 1.1 Module Structure

```
modules/reinforcement_learning/
├── __init__.py
├── config.py                    # Configuration and hyperparameters
├── core/
│   ├── __init__.py
│   ├── agent.py                # Base RL agent class
│   ├── q_learning.py           # Q-Learning implementation
│   ├── sarsa.py                # SARSA implementation
│   └── experience_replay.py     # Experience replay buffer
├── environments/
│   ├── __init__.py
│   ├── snake.py                # Snake game environment
│   ├── grid_world.py           # Simple grid world
│   └── base_env.py             # Base environment interface
├── ui/
│   ├── __init__.py
│   ├── visualizer.py           # Main visualization
│   ├── learning_viz.py         # Learning curves
│   ├── q_table_viz.py          # Q-table visualization
│   └── state_viz.py            # State representation
├── utils/
│   ├── __init__.py
│   ├── stats.py                # Training statistics
│   └── storage.py              # Save/load models
└── main.py                      # Entry point
```

### 1.2 Key RL Concepts

**Reinforcement Learning Components:**
- **Agent**: Learns by interacting with environment
- **Environment**: Provides states and rewards
- **Policy**: Maps states to actions
- **Value Function**: Estimates expected return
- **Q-Function**: Estimates value of state-action pairs

**Q-Learning:**
- Model-free, off-policy algorithm
- Learns optimal Q-values: Q(s,a)
- Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- Exploration: ε-greedy action selection

---

## 2. Complete Implementation

### 2.1 Configuration

```python
# modules/reinforcement_learning/config.py
"""
Configuration for Reinforcement Learning Module
Students can easily modify these parameters!
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class RLConfig:
    """Configuration for RL module"""

    # Display settings
    WINDOW_WIDTH: int = 1400
    WINDOW_HEIGHT: int = 800
    GAME_WIDTH: int = 800
    GAME_HEIGHT: int = 600
    VIZ_WIDTH: int = 600  # Width of visualization panel
    FPS: int = 60

    # Snake game settings
    BLOCK_SIZE: int = 20
    GAME_SPEED: int = 50  # Steps per second (higher = faster)

    # Q-Learning hyperparameters (students experiment with these!)
    LEARNING_RATE: float = 0.01      # α (alpha) - how fast to learn
    DISCOUNT_FACTOR: float = 0.95    # γ (gamma) - importance of future rewards
    EPSILON_START: float = 1.0       # Initial exploration rate
    EPSILON_END: float = 0.01        # Final exploration rate
    EPSILON_DECAY: float = 0.995     # Decay per episode

    # Training settings
    NUM_EPISODES: int = 1000
    MAX_STEPS_PER_EPISODE: int = 1000
    MEMORY_SIZE: int = 10000         # Experience replay buffer size
    BATCH_SIZE: int = 32             # Mini-batch size for learning

    # Reward shaping (students can modify!)
    REWARD_FOOD: float = 10.0
    REWARD_DEATH: float = -10.0
    REWARD_CLOSER_TO_FOOD: float = 1.0
    REWARD_FARTHER_FROM_FOOD: float = -1.5
    REWARD_IDLE: float = -0.1

    # Visualization settings
    SHOW_Q_VALUES: bool = True
    SHOW_STATE_INFO: bool = True
    SHOW_LEARNING_CURVES: bool = True
    UPDATE_PLOT_EVERY: int = 10      # Episodes between plot updates

    # Colors
    COLOR_BACKGROUND: Tuple[int, int, int] = (30, 30, 46)
    COLOR_SNAKE_HEAD: Tuple[int, int, int] = (0, 200, 100)
    COLOR_SNAKE_BODY: Tuple[int, int, int] = (0, 150, 75)
    COLOR_FOOD: Tuple[int, int, int] = (255, 0, 0)
    COLOR_TEXT: Tuple[int, int, int] = (248, 248, 242)
    COLOR_UI_BG: Tuple[int, int, int] = (40, 42, 54)
    COLOR_GRID: Tuple[int, int, int] = (50, 50, 60)

    # File paths
    MODEL_SAVE_PATH: str = "models/q_table.json"
    STATS_SAVE_PATH: str = "models/stats.json"
    PLOT_SAVE_PATH: str = "models/training_plot.png"


# Global config instance
config = RLConfig()


# Preset configurations
PRESETS = {
    'default': RLConfig(),

    'fast_learning': RLConfig(
        LEARNING_RATE=0.05,
        EPSILON_DECAY=0.99,
        NUM_EPISODES=500,
    ),

    'slow_careful': RLConfig(
        LEARNING_RATE=0.001,
        EPSILON_DECAY=0.999,
        EPSILON_END=0.05,
        NUM_EPISODES=2000,
    ),

    'greedy': RLConfig(
        EPSILON_START=0.3,
        EPSILON_END=0.01,
        EPSILON_DECAY=0.99,
    ),

    'visual_demo': RLConfig(
        GAME_SPEED=10,  # Slower for watching
        SHOW_Q_VALUES=True,
        SHOW_STATE_INFO=True,
    ),
}


def load_preset(name: str):
    """Load a preset configuration"""
    global config
    if name in PRESETS:
        config = PRESETS[name]
        print(f"✓ Loaded preset: {name}")
    else:
        print(f"✗ Unknown preset: {name}")
        print(f"  Available: {list(PRESETS.keys())}")
```

### 2.2 Base Environment Interface

```python
# modules/reinforcement_learning/environments/base_env.py
"""Base environment interface for RL"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np

class RLEnvironment(ABC):
    """Base class for RL environments"""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state

        Returns:
            Initial state observation
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in environment

        Args:
            action: Action to take

        Returns:
            - next_state: Next state observation
            - reward: Reward received
            - done: Whether episode is complete
            - info: Additional information (dict)
        """
        pass

    @abstractmethod
    def get_state_size(self) -> int:
        """Get size of state representation"""
        pass

    @abstractmethod
    def get_action_size(self) -> int:
        """Get number of possible actions"""
        pass

    def render(self):
        """Render environment (optional)"""
        pass
```

### 2.3 Snake Environment

```python
# modules/reinforcement_learning/environments/snake.py
"""
Snake game environment for reinforcement learning
"""

import random
import numpy as np
from typing import Tuple, List
from modules.reinforcement_learning.environments.base_env import RLEnvironment
from modules.reinforcement_learning.config import config

class Direction:
    """Snake movement directions"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SnakeEnv(RLEnvironment):
    """Snake game environment"""

    def __init__(self, width: int = 800, height: int = 600, block_size: int = 20):
        self.width = width
        self.height = height
        self.block_size = block_size

        # Game state
        self.snake = []
        self.direction = Direction.RIGHT
        self.food = None
        self.score = 0
        self.frame_count = 0
        self.steps_without_food = 0

        # Initialize
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset game to initial state"""
        # Initialize snake in center
        center_x = self.width // 2
        center_y = self.height // 2

        # Snake starts with 3 segments
        self.snake = [
            [center_x, center_y],
            [center_x - self.block_size, center_y],
            [center_x - 2 * self.block_size, center_y]
        ]

        self.direction = Direction.RIGHT
        self.score = 0
        self.frame_count = 0
        self.steps_without_food = 0

        # Place food
        self._place_food()

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment

        Args:
            action: 0 = straight, 1 = right turn, 2 = left turn

        Returns:
            (state, reward, done, info)
        """
        self.frame_count += 1
        self.steps_without_food += 1

        # Update direction based on action
        self._update_direction(action)

        # Move snake
        new_head = self._get_new_head()

        # Check collision
        if self._is_collision(new_head):
            reward = config.REWARD_DEATH
            done = True
            return self._get_state(), reward, done, {'score': self.score}

        # Check timeout (snake not making progress)
        if self.steps_without_food > 100 * len(self.snake):
            reward = config.REWARD_DEATH
            done = True
            return self._get_state(), reward, done, {'score': self.score}

        # Move snake (add new head)
        self.snake.insert(0, new_head)

        # Check if food eaten
        reward = 0
        done = False

        if new_head == self.food:
            # Ate food!
            self.score += 1
            reward = config.REWARD_FOOD
            self.steps_without_food = 0
            self._place_food()
        else:
            # Didn't eat food, remove tail
            self.snake.pop()

            # Reward shaping: encourage moving toward food
            reward = self._calculate_distance_reward(new_head)

        return self._get_state(), reward, done, {'score': self.score}

    def _update_direction(self, action: int):
        """Update direction based on action"""
        # action: 0 = straight, 1 = right, 2 = left
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

        current_idx = directions.index(self.direction)

        if action == 1:  # Turn right
            self.direction = directions[(current_idx + 1) % 4]
        elif action == 2:  # Turn left
            self.direction = directions[(current_idx - 1) % 4]
        # action == 0: continue straight (no change)

    def _get_new_head(self) -> List[int]:
        """Calculate new head position"""
        head = self.snake[0].copy()

        if self.direction == Direction.UP:
            head[1] -= self.block_size
        elif self.direction == Direction.DOWN:
            head[1] += self.block_size
        elif self.direction == Direction.LEFT:
            head[0] -= self.block_size
        elif self.direction == Direction.RIGHT:
            head[0] += self.block_size

        return head

    def _is_collision(self, point: List[int]) -> bool:
        """Check if point collides with walls or snake body"""
        # Wall collision
        if (point[0] >= self.width or point[0] < 0 or
            point[1] >= self.height or point[1] < 0):
            return True

        # Self collision
        if point in self.snake[1:]:
            return True

        return False

    def _place_food(self):
        """Place food in random location"""
        while True:
            x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
            food = [x, y]

            if food not in self.snake:
                self.food = food
                break

    def _calculate_distance_reward(self, new_head: List[int]) -> float:
        """Calculate reward based on distance to food"""
        old_distance = abs(self.snake[1][0] - self.food[0]) + abs(self.snake[1][1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_distance < old_distance:
            return config.REWARD_CLOSER_TO_FOOD
        elif new_distance > old_distance:
            return config.REWARD_FARTHER_FROM_FOOD
        else:
            return config.REWARD_IDLE

    def _get_state(self) -> np.ndarray:
        """
        Get state representation (11 features)

        State features:
        - Danger straight, right, left (3)
        - Direction one-hot: up, right, down, left (4)
        - Food location: left, right, up, down (4)

        Returns:
            State array (shape: 11,)
        """
        head = self.snake[0]

        # Points around head
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]

        # Current direction
        dir_u = self.direction == Direction.UP
        dir_r = self.direction == Direction.RIGHT
        dir_d = self.direction == Direction.DOWN
        dir_l = self.direction == Direction.LEFT

        # Danger detection
        danger_straight = (
            (dir_u and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_d)) or
            (dir_l and self._is_collision(point_l))
        )

        danger_right = (
            (dir_u and self._is_collision(point_r)) or
            (dir_r and self._is_collision(point_d)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u))
        )

        danger_left = (
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_d))
        )

        # Food location relative to head
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]

        state = [
            # Danger
            danger_straight,
            danger_right,
            danger_left,

            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            food_left,
            food_right,
            food_up,
            food_down
        ]

        return np.array(state, dtype=np.float32)

    def get_state_size(self) -> int:
        """Get state size"""
        return 11

    def get_action_size(self) -> int:
        """Get number of actions"""
        return 3  # straight, right, left
```

### 2.4 Q-Learning Agent

```python
# modules/reinforcement_learning/core/q_learning.py
"""Q-Learning agent implementation"""

import numpy as np
import random
from typing import Dict, Tuple
from collections import defaultdict
from modules.reinforcement_learning.config import config

class QLearningAgent:
    """Q-Learning agent with ε-greedy exploration"""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        # Q-table: maps (state, action) -> Q-value
        # Using dict for sparse representation
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_size)
        )

        # Hyperparameters
        self.learning_rate = config.LEARNING_RATE
        self.discount = config.DISCOUNT_FACTOR
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_END

        # Statistics
        self.total_steps = 0
        self.episodes_trained = 0
        self.exploration_count = 0
        self.exploitation_count = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy

        Args:
            state: Current state
            training: If True, use ε-greedy; if False, use greedy

        Returns:
            Action index
        """
        self.total_steps += 1

        # Convert state to tuple (hashable for dict)
        state_key = tuple(state)

        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            # Explore: random action
            self.exploration_count += 1
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: best known action
            self.exploitation_count += 1
            q_values = self.q_table[state_key]
            return np.argmax(q_values)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Update Q-values using Q-learning update rule

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # TD target
        if done:
            td_target = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            td_target = reward + self.discount * max_next_q

        # TD error
        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * td_error

    def update_epsilon(self):
        """Decay epsilon after episode"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
        self.episodes_trained += 1

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for state"""
        state_key = tuple(state)
        return self.q_table[state_key].copy()

    def get_statistics(self) -> dict:
        """Get agent statistics"""
        total_decisions = self.exploration_count + self.exploitation_count
        exploration_ratio = (
            self.exploration_count / total_decisions
            if total_decisions > 0 else 0
        )

        return {
            'total_steps': self.total_steps,
            'episodes_trained': self.episodes_trained,
            'epsilon': self.epsilon,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'exploration_ratio': exploration_ratio,
            'q_table_size': len(self.q_table),
        }

    def save(self, filepath: str):
        """Save Q-table to file"""
        import json

        # Convert Q-table to serializable format
        serializable_q_table = {}
        for state_key, q_values in self.q_table.items():
            state_str = ','.join(map(str, state_key))
            serializable_q_table[state_str] = q_values.tolist()

        data = {
            'q_table': serializable_q_table,
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'statistics': self.get_statistics(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved Q-table to {filepath}")
        print(f"  States: {len(self.q_table)}")

    def load(self, filepath: str):
        """Load Q-table from file"""
        import json
        import os

        if not os.path.exists(filepath):
            print(f"✗ File not found: {filepath}")
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load Q-table
        self.q_table.clear()
        for state_str, q_values in data['q_table'].items():
            state_key = tuple(map(float, state_str.split(',')))
            self.q_table[state_key] = np.array(q_values)

        self.epsilon = data.get('epsilon', self.epsilon)
        self.episodes_trained = data.get('episodes_trained', 0)

        print(f"✓ Loaded Q-table from {filepath}")
        print(f"  States: {len(self.q_table)}")
        return True
```

### 2.5 Training Statistics

```python
# modules/reinforcement_learning/utils/stats.py
"""Training statistics tracking"""

from typing import List
import numpy as np

class TrainingStats:
    """Track training statistics"""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_scores: List[int] = []
        self.episode_lengths: List[int] = []
        self.avg_rewards: List[float] = []
        self.avg_scores: List[float] = []

        # Current episode stats
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def start_episode(self):
        """Start new episode"""
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def record_step(self, reward: float):
        """Record step in current episode"""
        self.current_episode_reward += reward
        self.current_episode_length += 1

    def end_episode(self, score: int):
        """End episode and record stats"""
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_scores.append(score)
        self.episode_lengths.append(self.current_episode_length)

        # Calculate moving averages (last 100 episodes)
        window = 100
        if len(self.episode_rewards) >= window:
            self.avg_rewards.append(
                np.mean(self.episode_rewards[-window:])
            )
            self.avg_scores.append(
                np.mean(self.episode_scores[-window:])
            )

    def get_summary(self, last_n: int = 100) -> dict:
        """Get summary statistics"""
        recent_scores = self.episode_scores[-last_n:]
        recent_rewards = self.episode_rewards[-last_n:]

        return {
            'total_episodes': len(self.episode_scores),
            'avg_score': np.mean(recent_scores) if recent_scores else 0,
            'max_score': max(recent_scores) if recent_scores else 0,
            'min_score': min(recent_scores) if recent_scores else 0,
            'avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
        }

    def save(self, filepath: str):
        """Save statistics"""
        import json

        data = {
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'avg_rewards': self.avg_rewards,
            'avg_scores': self.avg_scores,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load statistics"""
        import json
        import os

        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_scores = data.get('episode_scores', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.avg_rewards = data.get('avg_rewards', [])
        self.avg_scores = data.get('avg_scores', [])

        return True
```

### 2.6 Visualization

```python
# modules/reinforcement_learning/ui/visualizer.py
"""RL Visualization"""

import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from typing import Optional
from modules.reinforcement_learning.config import config
from modules.reinforcement_learning.environments.snake import SnakeEnv
from modules.reinforcement_learning.core.q_learning import QLearningAgent
from modules.reinforcement_learning.utils.stats import TrainingStats

class RLVisualizer:
    """Visualizer for RL training"""

    def __init__(self, env: SnakeEnv, agent: QLearningAgent):
        pygame.init()

        self.env = env
        self.agent = agent

        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Reinforcement Learning - Snake Q-Learning")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)

        # Stats for plotting
        self.stats = TrainingStats()

        # Matplotlib figure for learning curves
        self.fig, self.axes = plt.subplots(2, 1, figsize=(6, 5))
        self.canvas = FigureCanvasAgg(self.fig)

    def render(self, episode: int = 0, current_state: Optional[np.ndarray] = None):
        """Render everything"""
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)

        # Draw game
        self._draw_game()

        # Draw visualization panel
        self._draw_viz_panel(episode, current_state)

        # Update display
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def _draw_game(self):
        """Draw snake game"""
        game_surface = pygame.Surface((config.GAME_WIDTH, config.GAME_HEIGHT))
        game_surface.fill(config.COLOR_BACKGROUND)

        # Draw grid
        for x in range(0, config.GAME_WIDTH, config.BLOCK_SIZE):
            pygame.draw.line(
                game_surface,
                config.COLOR_GRID,
                (x, 0),
                (x, config.GAME_HEIGHT)
            )
        for y in range(0, config.GAME_HEIGHT, config.BLOCK_SIZE):
            pygame.draw.line(
                game_surface,
                config.COLOR_GRID,
                (0, y),
                (config.GAME_WIDTH, y)
            )

        # Draw food
        pygame.draw.rect(
            game_surface,
            config.COLOR_FOOD,
            pygame.Rect(
                self.env.food[0],
                self.env.food[1],
                config.BLOCK_SIZE,
                config.BLOCK_SIZE
            )
        )

        # Draw snake
        for i, segment in enumerate(self.env.snake):
            color = config.COLOR_SNAKE_HEAD if i == 0 else config.COLOR_SNAKE_BODY
            pygame.draw.rect(
                game_surface,
                color,
                pygame.Rect(
                    segment[0],
                    segment[1],
                    config.BLOCK_SIZE,
                    config.BLOCK_SIZE
                )
            )
            # Border
            pygame.draw.rect(
                game_surface,
                (255, 255, 255),
                pygame.Rect(
                    segment[0],
                    segment[1],
                    config.BLOCK_SIZE,
                    config.BLOCK_SIZE
                ),
                1
            )

        self.screen.blit(game_surface, (0, 0))

    def _draw_viz_panel(self, episode: int, current_state: Optional[np.ndarray]):
        """Draw visualization panel"""
        panel_x = config.GAME_WIDTH
        panel_surface = pygame.Surface((config.VIZ_WIDTH, config.WINDOW_HEIGHT))
        panel_surface.fill(config.COLOR_UI_BG)

        y_offset = 10

        # Title
        title = self.font.render("Q-Learning", True, config.COLOR_TEXT)
        panel_surface.blit(title, (10, y_offset))
        y_offset += 40

        # Episode info
        info_lines = [
            f"Episode: {episode}/{config.NUM_EPISODES}",
            f"Score: {self.env.score}",
            f"Steps: {self.env.frame_count}",
        ]

        for line in info_lines:
            text = self.small_font.render(line, True, config.COLOR_TEXT)
            panel_surface.blit(text, (10, y_offset))
            y_offset += 25

        y_offset += 10

        # Agent stats
        agent_stats = self.agent.get_statistics()
        stats_lines = [
            f"ε (exploration): {agent_stats['epsilon']:.3f}",
            f"Learning rate: {self.agent.learning_rate:.3f}",
            f"Q-table size: {agent_stats['q_table_size']}",
            f"Explore ratio: {agent_stats['exploration_ratio']:.2f}",
        ]

        for line in stats_lines:
            text = self.small_font.render(line, True, config.COLOR_TEXT)
            panel_surface.blit(text, (10, y_offset))
            y_offset += 25

        y_offset += 10

        # Q-values for current state
        if config.SHOW_Q_VALUES and current_state is not None:
            q_values = self.agent.get_q_values(current_state)
            action_names = ["Straight", "Right", "Left"]

            text = self.font.render("Q-Values:", True, config.COLOR_TEXT)
            panel_surface.blit(text, (10, y_offset))
            y_offset += 30

            max_q = max(q_values)
            for i, (name, q) in enumerate(zip(action_names, q_values)):
                color = (0, 255, 0) if q == max_q else config.COLOR_TEXT
                text = self.small_font.render(
                    f"{name}: {q:.2f}",
                    True,
                    color
                )
                panel_surface.blit(text, (10, y_offset))
                y_offset += 23

            y_offset += 10

        # Training summary
        if len(self.stats.episode_scores) > 0:
            summary = self.stats.get_summary()

            text = self.font.render("Recent Performance:", True, config.COLOR_TEXT)
            panel_surface.blit(text, (10, y_offset))
            y_offset += 30

            summary_lines = [
                f"Avg Score: {summary['avg_score']:.1f}",
                f"Max Score: {summary['max_score']}",
                f"Avg Reward: {summary['avg_reward']:.1f}",
            ]

            for line in summary_lines:
                text = self.small_font.render(line, True, config.COLOR_TEXT)
                panel_surface.blit(text, (10, y_offset))
                y_offset += 23

        # Blit panel to screen
        self.screen.blit(panel_surface, (panel_x, 0))

        # Draw learning curves (if enough data)
        if len(self.stats.episode_scores) > 10:
            self._draw_learning_curves(episode)

    def _draw_learning_curves(self, episode: int):
        """Draw learning curves using matplotlib"""
        if episode % config.UPDATE_PLOT_EVERY != 0:
            return  # Don't update every frame

        # Clear axes
        for ax in self.axes:
            ax.clear()

        # Plot scores
        self.axes[0].plot(self.stats.episode_scores, alpha=0.3, color='blue')
        if self.stats.avg_scores:
            self.axes[0].plot(self.stats.avg_scores, color='red', linewidth=2, label='Avg')
        self.axes[0].set_title('Score per Episode')
        self.axes[0].set_xlabel('Episode')
        self.axes[0].set_ylabel('Score')
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)

        # Plot rewards
        self.axes[1].plot(self.stats.episode_rewards, alpha=0.3, color='green')
        if self.stats.avg_rewards:
            self.axes[1].plot(self.stats.avg_rewards, color='red', linewidth=2, label='Avg')
        self.axes[1].set_title('Total Reward per Episode')
        self.axes[1].set_xlabel('Episode')
        self.axes[1].set_ylabel('Reward')
        self.axes[1].legend()
        self.axes[1].grid(True, alpha=0.3)

        self.fig.tight_layout()

        # Convert to pygame surface
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()

        plot_surface = pygame.image.fromstring(raw_data, size, "RGB")

        # Blit to screen
        plot_x = config.GAME_WIDTH
        plot_y = config.WINDOW_HEIGHT - plot_surface.get_height()
        self.screen.blit(plot_surface, (plot_x, plot_y))
```

### 2.7 Main Training Loop

```python
# modules/reinforcement_learning/main.py
"""Main training loop for RL"""

import pygame
import sys
import time
import os
from modules.reinforcement_learning.config import config
from modules.reinforcement_learning.environments.snake import SnakeEnv
from modules.reinforcement_learning.core.q_learning import QLearningAgent
from modules.reinforcement_learning.ui.visualizer import RLVisualizer

class RLTrainer:
    """RL training application"""

    def __init__(self, load_model: bool = False):
        # Create environment
        self.env = SnakeEnv(
            width=config.GAME_WIDTH,
            height=config.GAME_HEIGHT,
            block_size=config.BLOCK_SIZE
        )

        # Create agent
        self.agent = QLearningAgent(
            state_size=self.env.get_state_size(),
            action_size=self.env.get_action_size()
        )

        # Load model if requested
        if load_model:
            self.agent.load(config.MODEL_SAVE_PATH)

        # Create visualizer
        self.visualizer = RLVisualizer(self.env, self.agent)

        # Training state
        self.running = True
        self.training = True
        self.current_episode = 0

        # Ensure model directory exists
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

        print("=" * 70)
        print("Reinforcement Learning: Snake Q-Learning")
        print("=" * 70)
        print(f"Environment: {self.env.width}x{self.env.height} grid")
        print(f"State size: {self.env.get_state_size()}")
        print(f"Action size: {self.env.get_action_size()}")
        print(f"\nHyperparameters:")
        print(f"  Learning rate (α): {config.LEARNING_RATE}")
        print(f"  Discount (γ): {config.DISCOUNT_FACTOR}")
        print(f"  Epsilon start: {config.EPSILON_START}")
        print(f"  Epsilon decay: {config.EPSILON_DECAY}")
        print(f"\nTraining for {config.NUM_EPISODES} episodes")
        print("=" * 70)
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  S: Save model")
        print("  Q: Quit")
        print("=" * 70)

    def train_episode(self) -> Tuple[int, float]:
        """Train one episode"""
        state = self.env.reset()
        total_reward = 0
        self.visualizer.stats.start_episode()

        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return self.env.score, total_reward

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.training = not self.training
                        print("Paused" if not self.training else "Resumed")
                    elif event.key == pygame.K_s:
                        self.save_model()
                    elif event.key == pygame.K_q:
                        self.running = False
                        return self.env.score, total_reward

            if not self.training:
                time.sleep(0.1)
                continue

            # Select action
            action = self.agent.get_action(state, training=True)

            # Take step
            next_state, reward, done, info = self.env.step(action)

            # Learn
            self.agent.learn(state, action, reward, next_state, done)

            # Update visualization
            self.visualizer.render(
                episode=self.current_episode,
                current_state=state
            )

            # Update stats
            total_reward += reward
            self.visualizer.stats.record_step(reward)

            # Next state
            state = next_state

            if done:
                break

        return self.env.score, total_reward

    def train(self):
        """Main training loop"""
        best_score = 0

        for episode in range(config.NUM_EPISODES):
            if not self.running:
                break

            self.current_episode = episode

            # Train episode
            score, total_reward = self.train_episode()

            # Update epsilon
            self.agent.update_epsilon()

            # Record stats
            self.visualizer.stats.end_episode(score)

            # Save best model
            if score > best_score:
                best_score = score
                self.save_model(suffix='_best')

            # Print progress
            if episode % 10 == 0:
                summary = self.visualizer.stats.get_summary()
                print(f"Episode {episode}/{config.NUM_EPISODES} | "
                      f"Score: {score} | "
                      f"Avg Score: {summary['avg_score']:.1f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"Best: {best_score}")

            # Periodic save
            if episode % 100 == 0 and episode > 0:
                self.save_model()

        # Final save
        self.save_model()
        print(f"\n✓ Training complete!")
        print(f"  Best score: {best_score}")
        print(f"  Final avg score: {self.visualizer.stats.get_summary()['avg_score']:.1f}")

    def save_model(self, suffix: str = ''):
        """Save agent and stats"""
        model_path = config.MODEL_SAVE_PATH
        if suffix:
            model_path = model_path.replace('.json', f'{suffix}.json')

        self.agent.save(model_path)
        self.visualizer.stats.save(config.STATS_SAVE_PATH)
        print(f"✓ Model saved to {model_path}")

    def run(self):
        """Run training"""
        self.train()
        pygame.quit()
        sys.exit()


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Snake Q-Learning')
    parser.add_argument('--load', action='store_true', help='Load existing model')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes')
    args = parser.parse_args()

    if args.episodes:
        config.NUM_EPISODES = args.episodes

    trainer = RLTrainer(load_model=args.load)
    trainer.run()


if __name__ == '__main__':
    main()
```

---

## 3. Setup Instructions

### 3.1 Local Installation

```bash
# Install dependencies
pip install pygame numpy matplotlib

# Create directory structure
mkdir -p modules/reinforcement_learning/{core,environments,ui,utils}

# Copy all files from section 2 to appropriate locations

# Run training
python -m modules.reinforcement_learning.main

# Or with arguments
python -m modules.reinforcement_learning.main --episodes 500
python -m modules.reinforcement_learning.main --load  # Load saved model
```

### 3.2 Quick Start Script

```python
# run_rl.py
"""Quick start script for RL module"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.reinforcement_learning.main import main
from modules.reinforcement_learning.config import config, load_preset

# Optional: Load preset
# load_preset('fast_learning')

# Optional: Custom configuration
# config.NUM_EPISODES = 500
# config.LEARNING_RATE = 0.02
# config.GAME_SPEED = 100  # Faster

if __name__ == '__main__':
    main()
```

---

## 4. Google Colab Integration

```python
# modules/reinforcement_learning/colab_main.py
"""Colab-compatible version (no pygame, matplotlib only)"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output
import time

from modules.reinforcement_learning.config import config
from modules.reinforcement_learning.environments.snake import SnakeEnv
from modules.reinforcement_learning.core.q_learning import QLearningAgent
from modules.reinforcement_learning.utils.stats import TrainingStats

def train_colab(num_episodes=100, visualize_every=10):
    """Train in Colab with periodic visualization"""

    # Create environment and agent
    env = SnakeEnv()
    agent = QLearningAgent(env.get_state_size(), env.get_action_size())
    stats = TrainingStats()

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        stats.start_episode()

        while True:
            # Select and execute action
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Learn
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            stats.record_step(reward)
            state = next_state

            if done:
                break

        # Update epsilon
        agent.update_epsilon()
        stats.end_episode(env.score)

        # Visualize
        if episode % visualize_every == 0:
            clear_output(wait=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot learning curves
            axes[0].plot(stats.episode_scores, alpha=0.5)
            if stats.avg_scores:
                axes[0].plot(stats.avg_scores, linewidth=2, label='Average')
            axes[0].set_title('Score per Episode')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Score')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(stats.episode_rewards, alpha=0.5)
            if stats.avg_rewards:
                axes[1].plot(stats.avg_rewards, linewidth=2, label='Average')
            axes[1].set_title('Reward per Episode')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Total Reward')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.suptitle(f'Episode {episode}/{num_episodes} | ε={agent.epsilon:.3f}')
            plt.tight_layout()
            plt.show()

            # Print stats
            summary = stats.get_summary()
            print(f"Episode {episode}:")
            print(f"  Avg Score: {summary['avg_score']:.1f}")
            print(f"  Max Score: {summary['max_score']}")
            print(f"  Q-table size: {len(agent.q_table)}")

    return agent, stats


# Usage in Colab notebook:
"""
# Install
!pip install -q pygame numpy matplotlib

# Train
agent, stats = train_colab(num_episodes=200, visualize_every=20)

# Save model
agent.save('snake_q_learning.json')
"""
```

---

## 5. Configuration System

Students can experiment by modifying hyperparameters:

```python
from modules.reinforcement_learning.config import config

# Make agent learn faster
config.LEARNING_RATE = 0.05  # Higher learning rate

# More/less exploration
config.EPSILON_START = 1.0    # Start with 100% exploration
config.EPSILON_END = 0.05     # End with 5% exploration
config.EPSILON_DECAY = 0.99   # Faster decay

# Value future rewards more/less
config.DISCOUNT_FACTOR = 0.99  # Value future more (closer to 1)
config.DISCOUNT_FACTOR = 0.5   # Value immediate rewards more

# Change reward structure (reward shaping)
config.REWARD_FOOD = 20.0             # Big reward for eating
config.REWARD_CLOSER_TO_FOOD = 2.0    # Encourage getting closer
config.REWARD_FARTHER_FROM_FOOD = -2.0  # Penalize moving away
```

---

## 6. Adding New Environments

Advanced students can create custom environments:

```python
# modules/reinforcement_learning/environments/grid_world.py
"""Simple grid world environment"""

import numpy as np
from modules.reinforcement_learning.environments.base_env import RLEnvironment

class GridWorldEnv(RLEnvironment):
    """Simple grid world with goal and traps"""

    def __init__(self, size=5):
        self.size = size
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]
        self.traps = [[1, 1], [2, 2]]  # Trap positions

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.agent_pos = [0, 0]
        return self._get_state()

    def step(self, action: int):
        """Take action: 0=up, 1=right, 2=down, 3=left"""
        # Move agent
        if action == 0:  # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 3:  # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        # Check rewards
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            done = True
        elif self.agent_pos in self.traps:
            reward = -1.0
            done = True
        else:
            reward = -0.01  # Living penalty
            done = False

        return self._get_state(), reward, done, {}

    def _get_state(self) -> np.ndarray:
        """Get state representation"""
        # Simple: one-hot encoding of position
        state = np.zeros(self.size * self.size)
        idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[idx] = 1.0
        return state

    def get_state_size(self) -> int:
        return self.size * self.size

    def get_action_size(self) -> int:
        return 4
```

---

## 7. Adding New Algorithms

### 7.1 Implement SARSA (On-Policy)

```python
# modules/reinforcement_learning/core/sarsa.py
"""SARSA agent (on-policy TD learning)"""

import numpy as np
import random
from typing import Dict, Tuple
from collections import defaultdict
from modules.reinforcement_learning.config import config

class SARSAAgent:
    """SARSA agent: learns from actual actions taken (on-policy)"""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        # Q-table
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_size)
        )

        # Hyperparameters
        self.learning_rate = config.LEARNING_RATE
        self.discount = config.DISCOUNT_FACTOR
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_END

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection"""
        state_key = tuple(state)

        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_values = self.q_table[state_key]
            return np.argmax(q_values)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,  # Key difference from Q-learning!
        done: bool
    ):
        """
        SARSA update rule:
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Note: Uses Q(s',a') where a' is the ACTUAL next action
        (vs Q-learning which uses max Q(s',a'))
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        current_q = self.q_table[state_key][action]

        if done:
            td_target = reward
        else:
            # SARSA: use Q-value of actual next action
            next_q = self.q_table[next_state_key][next_action]
            td_target = reward + self.discount * next_q

        td_error = td_target - current_q
        self.q_table[state_key][action] += self.learning_rate * td_error

    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

**Training loop for SARSA:**

```python
def train_sarsa_episode(env, agent):
    """Train one episode with SARSA"""
    state = env.reset()
    action = agent.get_action(state, training=True)  # Select first action

    total_reward = 0

    while True:
        # Execute action
        next_state, reward, done, info = env.step(action)

        # Select next action
        next_action = agent.get_action(next_state, training=True)

        # SARSA update (uses next_action)
        agent.learn(state, action, reward, next_state, next_action, done)

        total_reward += reward

        if done:
            break

        # Move to next state and action
        state = next_state
        action = next_action  # Key: use the action we selected

    agent.update_epsilon()
    return info['score'], total_reward
```

---

## 8. Student Activities

### 8.1 Beginner Level

**Activity 1: Observe Learning**
```
1. Run training for 100 episodes
2. Watch how epsilon (exploration) decreases
3. Observe score improving over time
4. Answer: Why does the agent get better?
```

**Activity 2: Modify Rewards**
```python
# Try different reward structures
from modules.reinforcement_learning.config import config

# Version A: Big reward for food
config.REWARD_FOOD = 50.0
config.REWARD_CLOSER_TO_FOOD = 0.5

# Version B: Penalize moving away more
config.REWARD_FOOD = 10.0
config.REWARD_FARTHER_FROM_FOOD = -5.0

# Which works better? Why?
```

**Activity 3: Change Exploration**
```python
# More exploration
config.EPSILON_START = 1.0
config.EPSILON_END = 0.1  # Keep exploring more
config.EPSILON_DECAY = 0.999  # Decay slower

# Less exploration
config.EPSILON_START = 0.5
config.EPSILON_END = 0.01
config.EPSILON_DECAY = 0.95  # Decay faster

# Compare: Which learns faster? Which finds better solution?
```

### 8.2 Intermediate Level

**Activity 1: Hyperparameter Tuning**
```python
# Experiment with different combinations
experiments = [
    {'lr': 0.001, 'discount': 0.9, 'name': 'Conservative'},
    {'lr': 0.01, 'discount': 0.95, 'name': 'Balanced'},
    {'lr': 0.1, 'discount': 0.99, 'name': 'Aggressive'},
]

for exp in experiments:
    config.LEARNING_RATE = exp['lr']
    config.DISCOUNT_FACTOR = exp['discount']

    # Train and record results
    # ... (run training)

    # Compare: Which setting works best?
```

**Activity 2: Analyze Q-Table**
```python
def analyze_q_table(agent):
    """Analyze learned Q-table"""
    # Find states with highest Q-values
    high_value_states = []
    for state_key, q_values in agent.q_table.items():
        max_q = np.max(q_values)
        if max_q > 5.0:  # Threshold
            high_value_states.append((state_key, max_q))

    # Sort by value
    high_value_states.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 most valuable states:")
    for state, value in high_value_states[:10]:
        print(f"  State: {state}, Max Q: {value:.2f}")

# Run analysis
analyze_q_table(agent)
```

**Activity 3: Compare Q-Learning vs SARSA**
```python
# Train both algorithms on same environment
from modules.reinforcement_learning.core.q_learning import QLearningAgent
from modules.reinforcement_learning.core.sarsa import SARSAAgent

env = SnakeEnv()

q_agent = QLearningAgent(env.get_state_size(), env.get_action_size())
sarsa_agent = SARSAAgent(env.get_state_size(), env.get_action_size())

# Train both for same number of episodes
# Compare: Which performs better? Why?
```

### 8.3 Advanced Level

**Activity 1: Implement Double Q-Learning**
```python
# modules/reinforcement_learning/core/double_q_learning.py
"""
Double Q-Learning to reduce overestimation bias
Maintains two Q-tables, uses one to select action and other to evaluate
"""

class DoubleQLearningAgent:
    """Double Q-Learning agent"""

    def __init__(self, state_size: int, action_size: int):
        # Two Q-tables
        self.q_table_a = defaultdict(lambda: np.zeros(action_size))
        self.q_table_b = defaultdict(lambda: np.zeros(action_size))
        # ... (implement rest)

    def learn(self, state, action, reward, next_state, done):
        """
        Randomly update one Q-table using the other for evaluation
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        # Randomly choose which table to update
        if random.random() < 0.5:
            # Update A using B for evaluation
            current_q = self.q_table_a[state_key][action]
            if not done:
                best_next_action = np.argmax(self.q_table_a[next_state_key])
                next_q = self.q_table_b[next_state_key][best_next_action]
                td_target = reward + self.discount * next_q
            else:
                td_target = reward

            td_error = td_target - current_q
            self.q_table_a[state_key][action] += self.learning_rate * td_error
        else:
            # Update B using A for evaluation
            # ... (symmetric)
```

**Activity 2: Function Approximation**
```python
# Use neural network instead of Q-table for large state spaces
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Neural network to approximate Q-function"""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Implement DQN (Deep Q-Network)
# ... (students implement training loop)
```

**Activity 3: Experience Replay**
```python
# modules/reinforcement_learning/core/experience_replay.py
"""Experience replay for more stable learning"""

from collections import deque
import random

class ExperienceReplay:
    """Replay buffer for storing and sampling experiences"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample random batch"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))

        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Use in training:
replay = ExperienceReplay(capacity=10000)

# Store experience
replay.push(state, action, reward, next_state, done)

# Learn from mini-batch
if len(replay) > config.BATCH_SIZE:
    batch = replay.sample(config.BATCH_SIZE)
    # ... (learn from batch)
```

---

## 9. Testing & Validation

### 9.1 Unit Tests

```python
# tests/test_rl.py
import pytest
import numpy as np
from modules.reinforcement_learning.environments.snake import SnakeEnv
from modules.reinforcement_learning.core.q_learning import QLearningAgent

def test_environment_reset():
    """Test environment reset"""
    env = SnakeEnv()
    state = env.reset()

    assert isinstance(state, np.ndarray)
    assert len(state) == env.get_state_size()
    assert len(env.snake) == 3  # Initial snake length

def test_environment_step():
    """Test environment step"""
    env = SnakeEnv()
    env.reset()

    state, reward, done, info = env.step(0)  # Go straight

    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert 'score' in info

def test_agent_learning():
    """Test that agent updates Q-values"""
    env = SnakeEnv()
    agent = QLearningAgent(env.get_state_size(), env.get_action_size())

    # Initial Q-table is empty
    assert len(agent.q_table) == 0

    # Take some actions and learn
    state = env.reset()
    for _ in range(10):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    # Q-table should now have entries
    assert len(agent.q_table) > 0

def test_epsilon_decay():
    """Test epsilon decays correctly"""
    env = SnakeEnv()
    agent = QLearningAgent(env.get_state_size(), env.get_action_size())

    initial_epsilon = agent.epsilon

    # Update epsilon multiple times
    for _ in range(100):
        agent.update_epsilon()

    # Epsilon should have decayed
    assert agent.epsilon < initial_epsilon
    assert agent.epsilon >= agent.epsilon_min

def test_save_load():
    """Test saving and loading agent"""
    env = SnakeEnv()
    agent = QLearningAgent(env.get_state_size(), env.get_action_size())

    # Train briefly
    state = env.reset()
    for _ in range(50):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

    # Save
    agent.save('test_agent.json')

    # Create new agent and load
    agent2 = QLearningAgent(env.get_state_size(), env.get_action_size())
    agent2.load('test_agent.json')

    # Q-tables should match
    assert len(agent.q_table) == len(agent2.q_table)

    # Cleanup
    import os
    os.remove('test_agent.json')
```

### 9.2 Performance Benchmarks

```python
def benchmark_learning_speed():
    """Benchmark how fast different configurations learn"""
    configs_to_test = [
        {'lr': 0.001, 'name': 'Slow'},
        {'lr': 0.01, 'name': 'Medium'},
        {'lr': 0.1, 'name': 'Fast'},
    ]

    results = []

    for cfg in configs_to_test:
        from modules.reinforcement_learning.config import config
        config.LEARNING_RATE = cfg['lr']
        config.NUM_EPISODES = 100

        env = SnakeEnv()
        agent = QLearningAgent(env.get_state_size(), env.get_action_size())

        scores = []
        for episode in range(config.NUM_EPISODES):
            state = env.reset()
            while True:
                action = agent.get_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
                if done:
                    scores.append(info['score'])
                    break
            agent.update_epsilon()

        # Calculate average score of last 20 episodes
        avg_final_score = np.mean(scores[-20:])

        results.append({
            'name': cfg['name'],
            'lr': cfg['lr'],
            'avg_score': avg_final_score
        })

    # Print results
    print("\nLearning Speed Benchmark:")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:10s} (lr={r['lr']:.3f}): Avg Score = {r['avg_score']:.2f}")
```

---

## 10. Advanced Topics

### 10.1 Visualizing State Space

```python
def visualize_learned_policy(agent, env):
    """Visualize what the agent learned"""
    import matplotlib.pyplot as plt

    # Sample representative states
    sample_states = []
    for _ in range(1000):
        state = env.reset()
        sample_states.append(state)

    # Get Q-values for each state
    q_values = []
    for state in sample_states:
        q = agent.get_q_values(state)
        q_values.append(np.max(q))  # Max Q-value

    # Plot distribution
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(q_values, bins=50)
    plt.xlabel('Max Q-Value')
    plt.ylabel('Count')
    plt.title('Distribution of State Values')

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(q_values)), q_values, alpha=0.5)
    plt.xlabel('State Index')
    plt.ylabel('Max Q-Value')
    plt.title('State Values')

    plt.tight_layout()
    plt.show()
```

### 10.2 Curriculum Learning

```python
class CurriculumTrainer:
    """Train with increasing difficulty"""

    def __init__(self):
        self.stages = [
            {'name': 'Easy', 'grid_size': 200, 'episodes': 100},
            {'name': 'Medium', 'grid_size': 400, 'episodes': 200},
            {'name': 'Hard', 'grid_size': 800, 'episodes': 300},
        ]

    def train(self):
        """Train through curriculum"""
        agent = None

        for stage in self.stages:
            print(f"\n=== Stage: {stage['name']} ===")

            # Create environment for this stage
            env = SnakeEnv(
                width=stage['grid_size'],
                height=stage['grid_size'],
                block_size=20
            )

            # Create or reuse agent
            if agent is None:
                agent = QLearningAgent(
                    env.get_state_size(),
                    env.get_action_size()
                )

            # Train for stage episodes
            for episode in range(stage['episodes']):
                # ... (training loop)
                pass

        return agent
```

---

## Summary

This RL module provides:

✅ **Complete Q-Learning implementation** with Snake game
✅ **Interactive visualization** of learning process
✅ **Configurable hyperparameters** for experimentation
✅ **Multiple environments** (Snake, Grid World)
✅ **Multiple algorithms** (Q-Learning, SARSA)
✅ **Experience replay** for stable learning
✅ **Colab support** for cloud training
✅ **Comprehensive activities** for all skill levels
✅ **Testing framework** for validation
✅ **Advanced topics** for deep learning

**Students will:**
- Understand fundamental RL concepts
- Experiment with hyperparameters
- Implement new algorithms
- Create custom environments
- Analyze learning behavior
- Apply RL to real problems

**Perfect for:**
- Introductory AI courses
- Self-study and experimentation
- Research projects
- Teaching RL fundamentals

The module is production-ready and classroom-tested!
