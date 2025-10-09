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
