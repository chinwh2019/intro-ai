import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame


class Config:
    """Configuration settings for the Snake Q-Learning environment"""

    def __init__(self):
        # Display settings
        self.WINDOW_WIDTH = 640
        self.WINDOW_HEIGHT = 480
        self.BLOCK_SIZE = 20
        self.SPEED = 50

        self.WINDOW_WIDTH = int(self.WINDOW_WIDTH)
        self.WINDOW_HEIGHT = int(self.WINDOW_HEIGHT)
        self.BLOCK_SIZE = int(self.BLOCK_SIZE)

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)

        # Training settings
        self.LEARNING_RATE = 0.01
        self.GAMMA = 0.95
        self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.995
        self.EPSILON_MIN = 0.01
        self.BATCH_SIZE = 32
        self.MEMORY_SIZE = 100000

        # File settings
        self.SAVE_DIR = "models"
        self.Q_TABLE_FILE = os.path.join(self.SAVE_DIR, "q_table.json")
        self.STATS_FILE = os.path.join(self.SAVE_DIR, "training_stats.json")
        self.PLOT_FILE = os.path.join(self.SAVE_DIR, "training_plot.png")

        # Create save directory if it doesn't exist
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)


class LearningVisualizer:
    """Real-time visualization of the learning process"""
    def __init__(self, config: Config, width=400):  # Accept width parameter
        self.config = config
        self.width = width
        self.height = 400
        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.Font(None, 24)
        self.action_names = ['Straight', 'Right Turn', 'Left Turn']
        self.total_reward = 0  # Add total reward tracking

        
    def update(self, agent, state: np.ndarray, action: int, reward: float):
        """Update the visualization with current learning data"""
        try:
            self.surface.fill(self.config.BLACK)
            
            # Update total reward
            self.total_reward += reward
            
            # Section 1: Title
            y_offset = 10
            title = "Q-Learning Visualization"
            self._draw_centered_text(title, y_offset, self.config.YELLOW)
            y_offset += 30
            
            # Section 2: Current Action and Rewards
            self._draw_text(f"Action: {self.action_names[action]}", (10, y_offset), 
                          color=self.config.GREEN)
            y_offset += 25
            
            # Show both current and total rewards
            self._draw_text("Current Reward:", (10, y_offset), self.config.GRAY)
            self._draw_text(f"{reward:.1f}", (140, y_offset), 
                          self.config.RED if reward < 0 else self.config.GREEN)
            y_offset += 25
            
            self._draw_text("Total Reward:", (10, y_offset), self.config.GRAY)
            self._draw_text(f"{self.total_reward:.1f}", (140, y_offset),
                          self.config.RED if self.total_reward < 0 else self.config.GREEN)
            y_offset += 40
            
            # Section 3: Q-values
            self._draw_text("Q-Values:", (10, y_offset))
            y_offset += 25
            
            # Draw Q-value bars
            bar_width = min(70, (self.width - 80) // 3)
            bar_spacing = 20
            bar_max_height = 80
            
            # Calculate total width of all bars
            total_bars_width = (bar_width + bar_spacing) * 3 - bar_spacing
            start_x = (self.width - total_bars_width) // 2
            
            # Get Q-values for current state
            q_values = agent.q_table.get(tuple(state), np.zeros(3))
            max_q = max(abs(max(q_values)), abs(min(q_values)), 1.0)
            
            for i, (action_name, q_val) in enumerate(zip(self.action_names, q_values)):
                normalized_height = (abs(float(q_val)) / max_q) * bar_max_height
                height = max(1, int(normalized_height))
                
                x = start_x + i * (bar_width + bar_spacing)
                y = y_offset + bar_max_height - height
                
                # Draw bar
                rect = pygame.Rect(int(x), int(y), int(bar_width), int(height))
                color = self.config.GREEN if i == action else self.config.BLUE
                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, self.config.WHITE, rect, 1)
                
                # Draw value
                value_text = f"{float(q_val):.2f}"
                self._draw_centered_text(value_text, y - 20, self.config.WHITE, x, bar_width)
                
                # Draw action name
                self._draw_centered_text(action_name, y_offset + bar_max_height + 10, 
                                      self.config.WHITE, x, bar_width)
            
            # Section 4: Agent Statistics
            y_offset += bar_max_height + 60
            
            # Draw statistics in a grid layout
            stats = [
                ("Epsilon", f"{agent.epsilon:.3f}"),
                ("Score", str(agent.best_score)),
                ("Exploration", str(agent.exploration_steps)),
                ("Exploitation", str(agent.exploitation_steps))
            ]
            
            # Calculate columns and spacing
            col_width = self.width // 2 - 20
            row_height = 25
            
            for i, (label, value) in enumerate(stats):
                row = i // 2
                col = i % 2
                x = 20 + col * col_width
                y = y_offset + row * row_height
                
                # Draw label and value with proper spacing
                label_surface = self.font.render(f"{label}:", True, self.config.GRAY)
                value_surface = self.font.render(str(value), True, self.config.WHITE)
                
                self.surface.blit(label_surface, (x, y))
                self.surface.blit(value_surface, (x + label_surface.get_width() + 10, y))
            
            return self.surface
            
        except Exception as e:
            print(f"Visualization error: {e}")
            traceback.print_exc()
            self.surface.fill(self.config.BLACK)
            self._draw_text("Visualization Error", (10, 10), self.config.RED)
            return self.surface
        
    def reset(self):
        """Reset total reward counter"""
        self.total_reward = 0
    
    def _draw_centered_text(self, text: str, y: int, color, x=None, width=None):
        """Draw text centered horizontally"""
        try:
            text_surface = self.font.render(str(text), True, color)
            if x is None:  # Center in entire surface
                x = (self.width - text_surface.get_width()) // 2
            elif width:  # Center in given width
                x = x + (width - text_surface.get_width()) // 2
            self.surface.blit(text_surface, (int(x), int(y)))
        except Exception as e:
            print(f"Text drawing error: {e}")
    
    def _draw_text(self, text: str, pos: tuple, color=None):
        """Draw text at specific position"""
        try:
            if color is None:
                color = self.config.WHITE
            text_surface = self.font.render(str(text), True, color)
            self.surface.blit(text_surface, (int(pos[0]), int(pos[1])))
        except Exception as e:
            print(f"Text drawing error: {e}")


class QValueVisualizer:
    """Visualization of Q-value updates"""

    def __init__(self, config: Config):
        self.config = config
        self.width = 300
        self.height = 200
        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.Font(None, 24)

    def visualize_update(
        self, old_q: float, new_q: float, reward: float, next_max_q: float
    ):
        """Visualize Q-value update process"""
        self.surface.fill(self.config.BLACK)

        # Draw update equation
        texts = [
            f"Old Q: {old_q:.2f}",
            f"Reward: {reward:.2f}",
            f"Next max Q: {next_max_q:.2f}",
            f"New Q: {new_q:.2f}",
            f"Difference: {new_q - old_q:.2f}",
        ]

        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, self.config.WHITE)
            self.surface.blit(text_surface, (10, 10 + i * 30))

        return self.surface


class StepVisualizer:
    """Visualization of Q-learning steps"""

    def __init__(self, config: Config):
        self.config = config
        self.width = 300
        self.height = 400
        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.Font(None, 24)
        self.steps = [
            "1. Observe State",
            "2. Choose Action (ε-greedy)",
            "3. Take Action",
            "4. Get Reward",
            "5. Update Q-value",
            "6. Move to Next State",
        ]
        self.current_step = 0

    def update(self, step: int, data: Dict):
        """Update the step visualization"""
        self.surface.fill(self.config.BLACK)
        self.current_step = step

        # Draw all steps
        for i, step_text in enumerate(self.steps):
            color = self.config.GREEN if i == step else self.config.GRAY
            text = self.font.render(step_text, True, color)
            self.surface.blit(text, (10, 10 + i * 30))

        # Draw step-specific information
        if data:
            self._draw_step_info(data)

        return self.surface

    def _draw_step_info(self, data: Dict):
        """Draw step-specific information"""
        y_position = 200
        for key, value in data.items():
            text = f"{key}: {value}"
            text_surface = self.font.render(text, True, self.config.WHITE)
            self.surface.blit(text_surface, (10, y_position))
            y_position += 25


class SnakeGameAI:
    """Enhanced Snake Game Environment with visualization and educational features"""

    def __init__(self, config: Config):
        pygame.init()
        self.config = config

        # Calculate total width to include visualization panels
        self.game_width = config.WINDOW_WIDTH
        self.visualization_width = 400  # Width for visualization panels
        self.total_width = self.game_width + self.visualization_width

        # Setup display
        self.screen = pygame.display.set_mode((self.total_width, config.WINDOW_HEIGHT))
        pygame.display.set_caption("Snake Q-Learning Visualization")
        self.clock = pygame.time.Clock()

        # Initialize visualizers
        self.learning_viz = LearningVisualizer(config, width=self.visualization_width)  # Pass width to visualizer

        # Direction mapping
        self.dir_to_move = {
            "RIGHT": [self.config.BLOCK_SIZE, 0],
            "LEFT": [-self.config.BLOCK_SIZE, 0],
            "UP": [0, -self.config.BLOCK_SIZE],
            "DOWN": [0, self.config.BLOCK_SIZE],
        }

        # Initialize game state
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset game state with proper initialization"""
        # Initialize snake position
        start_x = (
            (self.game_width // 2) // self.config.BLOCK_SIZE * self.config.BLOCK_SIZE
        )
        start_y = (
            (self.config.WINDOW_HEIGHT // 2)
            // self.config.BLOCK_SIZE
            * self.config.BLOCK_SIZE
        )

        self.snake = [
            [start_x, start_y],  # Head
            [start_x - self.config.BLOCK_SIZE, start_y],  # Body
            [start_x - (2 * self.config.BLOCK_SIZE), start_y],  # Tail
        ]

        self.direction = "RIGHT"
        self.score = 0
        self.food = self._place_food()
        self.frame_iteration = 0
        self.steps_without_food = 0
        self.total_reward = 0

        # Reset visualizer
        if hasattr(self, 'learning_viz'):
            self.learning_viz.reset()

        # Reset visualization states
        self.current_step = 0
        return self._get_state()

    def _place_food(self) -> List[int]:
        """Place food in random location, ensuring it's not on snake"""
        while True:
            x = (
                random.randrange(
                    0,
                    (self.game_width - self.config.BLOCK_SIZE)
                    // self.config.BLOCK_SIZE,
                )
                * self.config.BLOCK_SIZE
            )
            y = (
                random.randrange(
                    0,
                    (self.config.WINDOW_HEIGHT - self.config.BLOCK_SIZE)
                    // self.config.BLOCK_SIZE,
                )
                * self.config.BLOCK_SIZE
            )
            if [x, y] not in self.snake:
                return [x, y]

    def _get_state(self) -> np.ndarray:
        """Get state representation with normalized values"""
        head = self.snake[0]

        # Points around the head
        point_l = [head[0] - self.config.BLOCK_SIZE, head[1]]
        point_r = [head[0] + self.config.BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - self.config.BLOCK_SIZE]
        point_d = [head[0], head[1] + self.config.BLOCK_SIZE]

        # Current direction as one-hot encoding
        dir_u = float(self.direction == "UP")
        dir_r = float(self.direction == "RIGHT")
        dir_d = float(self.direction == "DOWN")
        dir_l = float(self.direction == "LEFT")

        # Normalized distances to food
        food_dist_x = (self.food[0] - head[0]) / self.game_width
        food_dist_y = (self.food[1] - head[1]) / self.config.WINDOW_HEIGHT

        state = [
            # Danger straight
            float(
                (dir_u and self._is_collision(point_u))
                or (dir_r and self._is_collision(point_r))
                or (dir_d and self._is_collision(point_d))
                or (dir_l and self._is_collision(point_l))
            ),
            # Danger right
            float(
                (dir_u and self._is_collision(point_r))
                or (dir_r and self._is_collision(point_d))
                or (dir_d and self._is_collision(point_l))
                or (dir_l and self._is_collision(point_u))
            ),
            # Danger left
            float(
                (dir_u and self._is_collision(point_l))
                or (dir_r and self._is_collision(point_u))
                or (dir_d and self._is_collision(point_r))
                or (dir_l and self._is_collision(point_d))
            ),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food direction
            float(self.food[0] < head[0]),  # food left
            float(self.food[0] > head[0]),  # food right
            float(self.food[1] < head[1]),  # food up
            float(self.food[1] > head[1]),  # food down
            # Normalized distances
            food_dist_x,
            food_dist_y,
        ]

        return np.array(state, dtype=np.float32)

    def _is_collision(self, point: List[int]) -> bool:
        """Check if point collides with walls or snake body"""
        return (
            point[0] >= self.game_width
            or point[0] < 0
            or point[1] >= self.config.WINDOW_HEIGHT
            or point[1] < 0
            or point in self.snake[1:]
        )

    def _calculate_reward(self, new_head: List[int], game_over: bool) -> float:
        """Calculate reward with improved shaping"""
        if game_over:
            return -10.0

        # Get current and new Manhattan distance to food
        current_dist = abs(self.snake[0][0] - self.food[0]) + abs(
            self.snake[0][1] - self.food[1]
        )
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        # Reward for eating food
        if new_head == self.food:
            self.steps_without_food = 0
            return 20.0

        # Reward for moving closer to food, penalty for moving away
        if new_dist < current_dist:
            return 1.0
        elif new_dist > current_dist:
            return -1.0

        return -0.1  # Small penalty for not making progress

    def step(self, action: int) -> Tuple[float, bool, int]:
        """Perform one step in the environment with visualization updates"""
        self.frame_iteration += 1
        self.steps_without_food += 1

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update direction
        clock_wise = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = clock_wise.index(self.direction)

        if action == 1:  # right turn
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:  # left turn
            new_dir = clock_wise[(idx - 1) % 4]
        else:  # continue straight
            new_dir = self.direction

        self.direction = new_dir

        # Move snake
        move = self.dir_to_move[self.direction]
        new_head = [self.snake[0][0] + move[0], self.snake[0][1] + move[1]]

        # Check for game over conditions
        game_over = False
        if self._is_collision(new_head) or self.steps_without_food > 100 * len(
            self.snake
        ):
            game_over = True

        # Calculate reward before updating snake
        reward = self._calculate_reward(new_head, game_over)

        if game_over:
            return reward, game_over, self.score

        # Update snake position
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()

        self.total_reward += reward
        return reward, game_over, self.score

    def render(self, agent=None, show_info: bool = True, action: int = None, reward: float = 0):
        """Render the game state with learning visualizations"""
        try:
            # Clear screen
            self.screen.fill(self.config.BLACK)

            # Draw game border
            pygame.draw.rect(
                self.screen,
                self.config.WHITE,
                pygame.Rect(0, 0, self.game_width, self.config.WINDOW_HEIGHT),
                2,
            )

            # Draw snake
            # Draw head
            head = self.snake[0]
            pygame.draw.rect(
                self.screen,
                self.config.GREEN,
                pygame.Rect(
                    int(head[0]),
                    int(head[1]),
                    self.config.BLOCK_SIZE,
                    self.config.BLOCK_SIZE,
                ),
            )

            # Draw body
            for segment in self.snake[1:]:
                pygame.draw.rect(
                    self.screen,
                    self.config.BLUE,
                    pygame.Rect(
                        int(segment[0]),
                        int(segment[1]),
                        self.config.BLOCK_SIZE,
                        self.config.BLOCK_SIZE,
                    ),
                )

                # Draw segment border
                pygame.draw.rect(
                    self.screen,
                    self.config.WHITE,
                    pygame.Rect(
                        int(segment[0]),
                        int(segment[1]),
                        self.config.BLOCK_SIZE,
                        self.config.BLOCK_SIZE,
                    ),
                    1,
                )

            # Draw food
            pygame.draw.rect(
                self.screen,
                self.config.RED,
                pygame.Rect(
                    int(self.food[0]),
                    int(self.food[1]),
                    self.config.BLOCK_SIZE,
                    self.config.BLOCK_SIZE,
                ),
            )

            # Draw visualizations
            if agent and show_info:
                try:
                    # Get current state
                    current_state = self._get_state()
                    
                    # Update and draw learning visualization
                    learning_surface = self.learning_viz.update(
                        agent,
                        current_state,
                        action if action is not None else 0,
                        float(reward)  # Ensure reward is float
                    )
                    if learning_surface:  # Check if surface was created successfully
                        self.screen.blit(learning_surface, (self.game_width + 10, 10))
                except Exception as viz_error:
                    print(f"Visualization error: {viz_error}")
                    traceback.print_exc()
            
            pygame.display.flip()
            
        except Exception as e:
            print(f"Render error: {e}")
            traceback.print_exc()


class ExperienceReplay:
    """Experience replay buffer for more stable learning"""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to memory"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """Sample random batch from memory"""
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self) -> int:
        return len(self.memory)


class QLearningAgent:
    """Enhanced Q-Learning Agent with visualization support and improved learning"""

    def __init__(self, config: Config, state_size: int = 14, action_size: int = 3):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size

        # Q-learning parameters
        self.q_table = {}
        self.learning_rate = config.LEARNING_RATE
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN

        # Experience replay
        self.memory = ExperienceReplay(config.MEMORY_SIZE)
        self.batch_size = config.BATCH_SIZE

        # Training statistics
        self.training_steps = 0
        self.exploration_steps = 0
        self.exploitation_steps = 0
        self.best_score = 0
        self.episode_rewards = []

        # Visualization data
        self.last_q_update = None

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get action using epsilon-greedy policy with visualization support"""
        state_key = tuple(state)

        # Initialize Q-values for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        # Exploration
        if training and random.random() < self.epsilon:
            self.exploration_steps += 1
            return random.randint(0, self.action_size - 1)

        # Exploitation
        self.exploitation_steps += 1
        return np.argmax(self.q_table[state_key])

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Update Q-values and store experience"""
        # Store experience
        self.memory.push(state, action, reward, next_state, done)

        # Convert states to tuples for dictionary keys
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        # Q-learning update
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_value = old_value + self.learning_rate * (
            reward + (0 if done else self.gamma * next_max) - old_value
        )

        self.q_table[state_key][action] = new_value

        # Store update information for visualization
        self.last_q_update = {
            "old_q": old_value,
            "new_q": new_value,
            "reward": reward,
            "next_max_q": next_max,
        }

        # Update training statistics
        self.training_steps += 1

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        """Learn from stored experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        for state, action, reward, next_state, done in batch:
            self.learn(state, action, reward, next_state, done)

    def save(self, filename: str = None):
        """Save agent state with additional information"""
        if filename is None:
            filename = self.config.Q_TABLE_FILE
            
        try:
            # Convert the Q-table to a serializable format
            serializable_q_table = {}
            for state, actions in self.q_table.items():
                state_str = ','.join([str(float(x)) for x in state])
                actions_list = [float(x) for x in actions]
                serializable_q_table[state_str] = actions_list

            data = {
                'q_table': serializable_q_table,
                'epsilon': float(self.epsilon),
                'training_steps': int(self.training_steps),
                'exploration_steps': int(self.exploration_steps),
                'exploitation_steps': int(self.exploitation_steps),
                'best_score': int(self.best_score),
                'episode_rewards': [float(x) for x in self.episode_rewards],
                'version': '1.0',  # Add version tracking
                'timestamp': str(datetime.now())
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f)
            print(f"Successfully saved agent state to {filename}")
            print(f"Q-table size: {len(self.q_table)} states")
            return True
        except Exception as e:
            print(f"Error saving agent state: {e}")
            return False

    def load(self, filename: str = None):
        """Load agent state with verification"""
        if filename is None:
            filename = self.config.Q_TABLE_FILE
            
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Convert loaded data to proper format
                self.q_table = {}
                for state_str, actions in data['q_table'].items():
                    state_values = [float(x) for x in state_str.split(',')]
                    state_tuple = tuple(state_values)
                    self.q_table[state_tuple] = np.array(actions, dtype=np.float32)
                
                self.epsilon = float(data.get('epsilon', self.config.EPSILON))
                self.training_steps = int(data.get('training_steps', 0))
                self.exploration_steps = int(data.get('exploration_steps', 0))
                self.exploitation_steps = int(data.get('exploitation_steps', 0))
                self.best_score = int(data.get('best_score', 0))
                self.episode_rewards = data.get('episode_rewards', [])
                
                print(f"Successfully loaded agent state from {filename}")
                print(f"Q-table size: {len(self.q_table)} states")
                print(f"Best score: {self.best_score}")
                return True
            return False
        except Exception as e:
            print(f"Error loading agent state: {e}")
            return False


class TrainingStats:
    """Track and analyze training performance"""

    def __init__(self):
        self.episode_scores = []
        self.episode_lengths = []
        self.episode_rewards = []
        self.moving_avg_size = 100

    def update(self, score: int, length: int, total_reward: float):
        """Update statistics with episode results"""
        self.episode_scores.append(score)
        self.episode_lengths.append(length)
        self.episode_rewards.append(total_reward)

    def get_moving_average(self, values: List[float]) -> float:
        """Calculate moving average of recent values"""
        if not values:
            return 0.0
        size = min(len(values), self.moving_avg_size)
        return sum(values[-size:]) / size

    def get_stats(self) -> Dict:
        """Get current training statistics"""
        return {
            "latest_score": self.episode_scores[-1] if self.episode_scores else 0,
            "avg_score": self.get_moving_average(self.episode_scores),
            "max_score": max(self.episode_scores) if self.episode_scores else 0,
            "avg_length": self.get_moving_average(self.episode_lengths),
            "avg_reward": self.get_moving_average(self.episode_rewards),
        }


def train(config: Config, load_existing: bool = True, episodes: int = 1000):
    """Enhanced training function with visualization and progress tracking"""
    env = SnakeGameAI(config)
    agent = QLearningAgent(config)
    stats = TrainingStats()

    if load_existing:
        if not agent.load():
            print("No existing model found or loading failed. Starting fresh training.")

    try:
        best_score = agent.best_score
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            current_action = None
            current_reward = 0

            while True:
                # Get and perform action
                action = agent.get_action(state)
                reward, done, score = env.step(action)
                next_state = env._get_state()

                # Store current action and reward for visualization
                current_action = action
                current_reward = reward

                # Learn from experience
                agent.learn(state, action, reward, next_state, done)

                # Perform experience replay
                if len(agent.memory) >= config.BATCH_SIZE:
                    agent.replay()

                episode_reward += reward
                steps += 1
                state = next_state

                # Update visualization
                env.render(
                    agent, show_info=True, action=current_action, reward=current_reward
                )
                env.clock.tick(config.SPEED)

                if done:
                    break

            # Update statistics
            stats.update(score, steps, episode_reward)
            current_stats = stats.get_stats()

            # Update best score
            if score > agent.best_score:
                best_score = score
                agent.best_score = best_score
                print(f"\nNew best score: {best_score}! Saving best model...")
                agent.save(config.Q_TABLE_FILE + ".best")

            # Save progress periodically
            if episode % 50 == 0:
                agent.save()

                # Print progress
                print(f"\nEpisode: {episode}")
                print(f"Score: {score}")
                print(f"Average Score: {current_stats['avg_score']:.2f}")
                print(f"Best Score: {agent.best_score}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Steps: {steps}")
                print(f"Total Reward: {episode_reward:.2f}")
                print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final state
        agent.save()

    return agent, stats


def play_trained_agent(config: Config, episodes: int = 5, delay: float = 0.1):
    """Demonstrate trained agent performance with detailed visualization"""
    env = SnakeGameAI(config)
    agent = QLearningAgent(config)
    
    # Try to load the best model first, if not available, load the regular model
    if os.path.exists(config.Q_TABLE_FILE + '.best'):
        if not agent.load(config.Q_TABLE_FILE + '.best'):
            print("Could not load best model, trying regular model...")
            if not agent.load():
                print("Could not load any trained model. Please train the agent first.")
                return
    else:
        if not agent.load():
            print("Could not load any trained model. Please train the agent first.")
            return
    
    print(f"\nLoaded Q-table with {len(agent.q_table)} states")
    print(f"Best score achieved during training: {agent.best_score}")
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0.0
    
    try:
        total_score = 0
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            while True:
                # Get action (no exploration during demonstration)
                state_key = tuple(state)
                if state_key in agent.q_table:
                    action = np.argmax(agent.q_table[state_key])
                else:
                    action = 0  # Default action if state not in Q-table
                
                reward, done, score = env.step(action)
                next_state = env._get_state()
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                # Render with additional information
                env.render(agent, show_info=True, action=action, reward=reward)
                env.clock.tick(config.SPEED)
                time.sleep(delay)
                
                if done:
                    break
            
            total_score += score
            print(f"Episode finished with score: {score}")
            print(f"Steps taken: {steps}")
            print(f"Total reward: {episode_reward:.2f}")
            time.sleep(1)
            
        print(f"\nDemonstration Summary:")
        print(f"Average Score: {total_score/episodes:.2f}")
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    finally:
        pygame.quit()

class InteractiveControls:
    """Interactive controls for adjusting training parameters"""

    def __init__(self, config: Config):
        self.config = config
        self.controls = {
            pygame.K_1: ("Learning Rate", 0.001, "learning_rate"),
            pygame.K_2: ("Epsilon", 0.1, "epsilon"),
            pygame.K_3: ("Speed", 5, "SPEED"),
        }
        self.font = pygame.font.Font(None, 24)

    def handle_input(self, event):
        """Handle keyboard input for parameter adjustment"""
        if event.type == pygame.KEYDOWN:
            if event.key in self.controls:
                name, delta, attr = self.controls[event.key]
                if event.mod & pygame.KMOD_SHIFT:  # Decrease if Shift is held
                    delta = -delta

                current = getattr(self.config, attr)
                new_value = max(0.001, current + delta)
                setattr(self.config, attr, new_value)
                print(f"{name} adjusted to: {new_value:.3f}")

    def draw(self, surface):
        """Draw control information"""
        text_lines = [
            "Controls:",
            "1: Adjust Learning Rate",
            "2: Adjust Epsilon",
            "3: Adjust Speed",
            "Hold Shift to decrease",
            f"Current LR: {self.config.learning_rate:.3f}",
            f"Current Epsilon: {self.config.epsilon:.3f}",
            f"Current Speed: {self.config.SPEED}",
        ]

        for i, line in enumerate(text_lines):
            text = self.font.render(line, True, self.config.WHITE)
            surface.blit(text, (10, 400 + i * 20))


def main():
    """Main program entry point with interactive menu"""
    parser = argparse.ArgumentParser(description="Snake Q-Learning Demo")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--speed", type=int, default=50, help="Game speed")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps")
    args = parser.parse_args()

    config = Config()
    config.SPEED = args.speed

    while True:
        print("\nSnake Q-Learning Demonstration")
        print("1. Train new agent")
        print("2. Continue training existing agent")
        print("3. Watch trained agent")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        try:
            if choice == "1":
                print("\nStarting new training session...")
                agent, stats = train(
                    config, load_existing=False, episodes=args.episodes
                )

            elif choice == "2":
                print("\nContinuing training from existing model...")
                agent, stats = train(config, load_existing=True, episodes=args.episodes)

            elif choice == "3":
                print("\nStarting demonstration of trained agent...")
                play_trained_agent(config, episodes=5, delay=args.delay)

            elif choice == "4":
                print("\nExiting program...")
                break

            else:
                print("Invalid choice! Please enter 1-4.")

        except KeyboardInterrupt:
            print("\nOperation interrupted by user")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            pygame.quit()


if __name__ == "__main__":
    # # Train new agent
    # python snake_qlearning.py --episodes 1000 --speed 50

    # Watch trained agent with slower speed
    # python snake_qlearning.py --episodes 5 --speed 30 --delay 0.2
    main()
