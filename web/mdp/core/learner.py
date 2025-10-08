"""Manual learning for MDP exploration"""

from typing import Dict, Tuple
from core.mdp import MDP, State


class MDPLearner:
    """Q-learning style updates for manual exploration"""

    @staticmethod
    def update_values(
        state: State,
        action: str,
        next_state: State,
        reward: float,
        V: Dict[State, float],
        Q: Dict[State, Dict[str, float]],
        discount: float,
        learning_rate: float
    ):
        """
        Update Q-value and V-value using Q-learning update rule

        Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]
        V(s) ← max Q(s,a)
        """
        # Ensure Q dictionary structure exists
        if state not in Q:
            Q[state] = {}
        if next_state not in Q:
            Q[next_state] = {}

        # Get current Q-value
        current_q = Q[state].get(action, 0.0)

        # Get max Q-value of next state
        if Q[next_state]:
            max_next_q = max(Q[next_state].values())
        else:
            max_next_q = 0.0

        # Q-learning update
        Q[state][action] = (1 - learning_rate) * current_q + \
                          learning_rate * (reward + discount * max_next_q)

        # Update V to max Q
        V[state] = max(Q[state].values()) if Q[state] else 0.0
