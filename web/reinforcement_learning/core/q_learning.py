"""Q-Learning agent implementation - Browser-safe (no NumPy)"""

import random
from typing import Dict, Tuple, List
from collections import defaultdict
from config import config

class QLearningAgent:
    """Q-Learning agent with ε-greedy exploration (pure Python)"""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        # Q-table: maps state_key -> list[float] of length action_size
        # Using dict with list instead of numpy array for browser compatibility
        self.q_table: Dict[Tuple, List[float]] = defaultdict(
            lambda: [0.0] * action_size
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

    def get_action(self, state, training: bool = True) -> int:
        """
        Select action using ε-greedy policy

        Args:
            state: Current state (tuple or list)
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
            # Find index of max value (pure Python)
            max_val = max(q_values)
            return q_values.index(max_val)

    def learn(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool
    ):
        """
        Update Q-values using Q-learning update rule

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        # Get Q-value lists
        q_values = self.q_table[state_key]
        next_q_values = self.q_table[next_state_key]

        # Current Q-value
        current_q = q_values[action]

        # TD target
        if done:
            td_target = reward
        else:
            max_next_q = max(next_q_values)
            td_target = reward + self.discount * max_next_q

        # TD error
        td_error = td_target - current_q

        # Update Q-value (in-place)
        q_values[action] = current_q + self.learning_rate * td_error

    def update_epsilon(self):
        """Decay epsilon after episode"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
        self.episodes_trained += 1

    def get_q_values(self, state) -> List[float]:
        """Get Q-values for state"""
        state_key = tuple(state)
        return list(self.q_table[state_key])  # Return copy as list

    def get_statistics(self) -> dict:
        """Get agent statistics"""
        total_decisions = self.exploration_count + self.exploitation_count
        exploration_ratio = (
            self.exploration_count / total_decisions
            if total_decisions > 0 else 0.0
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
        """Save Q-table to file (disabled in web version)"""
        print(f"\n⚠️ Save disabled in web version (browser storage not implemented)")
        print("  To save progress: use the desktop version (run_rl.py)")

    def load(self, filepath: str):
        """Load Q-table from file (disabled in web version)"""
        print(f"\n⚠️ Load disabled in web version (browser storage not implemented)")
        print("  To load models: use the desktop version (run_rl.py)")
        return False
