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
