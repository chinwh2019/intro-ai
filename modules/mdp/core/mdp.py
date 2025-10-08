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
