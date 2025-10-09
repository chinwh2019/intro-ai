"""Q-Learning agent implementation"""

import numpy as np
import random
from typing import Dict, Tuple
from collections import defaultdict
from config import config

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
        import os

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

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
