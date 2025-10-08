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

        # Set terminal state values based on their terminal rewards
        for state in mdp.terminal_states:
            if state in mdp.terminal_rewards:
                reward = mdp.terminal_rewards[state]
                self.values[state] = reward
                # Set Q-values for all actions to reward
                for action in mdp.actions:
                    self.q_values[(state, action)] = reward

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
                    # Keep terminal states at their reward values
                    if state in self.mdp.terminal_rewards:
                        self.values[state] = self.mdp.terminal_rewards[state]
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
