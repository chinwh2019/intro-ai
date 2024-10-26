import random
import numpy as np
from utils import DataManager


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.load_q_table()

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
            
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = [0, 0]
            
        return np.argmax(self.q_table[state_str])

    def update(self, state, action, reward, next_state):
        state_str = str(state)
        next_state_str = str(next_state)
        
        if state_str not in self.q_table:
            self.q_table[state_str] = [0, 0]
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = [0, 0]
            
        old_value = self.q_table[state_str][action]
        next_max = max(self.q_table[next_state_str])
        
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value)
        self.q_table[state_str][action] = new_value

    def save_q_table(self):
        DataManager.save_q_table(self.q_table)

    def load_q_table(self):
        self.q_table = DataManager.load_q_table()