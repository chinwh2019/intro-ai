"""Grid World environment for MDP"""

import random
from typing import Tuple, Set, List, Dict
from core.mdp import MDP, State
from config import config


class GridWorld:
    """Grid world environment"""

    def __init__(
        self,
        grid_size: int = 5,
        num_obstacles: int = 2,
        noise: float = 0.2,
        discount: float = 0.9,
        living_reward: float = -0.04
    ):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.noise = noise
        self.discount = discount
        self.living_reward = living_reward

        # Actions
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_effects = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }

        # Generate layout
        self.goal_pos = None
        self.danger_pos = None
        self.obstacles = set()
        self.start_pos = None

        self._generate_layout()

        # Build MDP
        self.mdp = self._build_mdp()

    def _generate_layout(self):
        """Generate random grid world layout"""
        # Place goal in top-right quadrant
        self.goal_pos = (
            random.randint(0, self.grid_size // 2),
            random.randint(self.grid_size // 2, self.grid_size - 1)
        )

        # Place danger adjacent to goal
        adjacent = [
            (self.goal_pos[0] - 1, self.goal_pos[1]),
            (self.goal_pos[0] + 1, self.goal_pos[1]),
            (self.goal_pos[0], self.goal_pos[1] - 1),
            (self.goal_pos[0], self.goal_pos[1] + 1),
        ]
        valid_danger = [
            pos for pos in adjacent
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
        ]
        self.danger_pos = random.choice(valid_danger) if valid_danger else (0, 0)

        # Place start in bottom-left quadrant
        while True:
            self.start_pos = (
                random.randint(self.grid_size // 2, self.grid_size - 1),
                random.randint(0, self.grid_size // 2)
            )
            if (self.start_pos != self.goal_pos and
                self.start_pos != self.danger_pos):
                break

        # Place obstacles
        all_positions = {
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        }
        forbidden = {self.start_pos, self.goal_pos, self.danger_pos}
        available = all_positions - forbidden

        self.obstacles = set(
            random.sample(list(available), min(self.num_obstacles, len(available)))
        )

    def _build_mdp(self) -> MDP:
        """Build MDP from grid world"""
        # States
        states = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.obstacles:
                    states.add(State((r, c)))

        start_state = State(self.start_pos)
        terminal_states = {State(self.goal_pos), State(self.danger_pos)}

        # Transitions and rewards
        transitions = {}
        rewards = {}

        for state in states:
            if state in terminal_states:
                continue

            for action in self.actions:
                # Get intended next state
                intended_next = self._get_next_position(state.position, action)

                # Build transition distribution (with noise)
                trans_dist = {}

                # Intended direction (1 - noise probability)
                intended_state = State(intended_next)
                if intended_state in states:
                    trans_dist[intended_state] = 1.0 - self.noise
                else:
                    trans_dist[state] = 1.0 - self.noise  # Hit wall, stay

                # Perpendicular directions (noise/2 each)
                perp_actions = self._get_perpendicular_actions(action)
                for perp_action in perp_actions:
                    perp_next = self._get_next_position(state.position, perp_action)
                    perp_state = State(perp_next)

                    if perp_state in states:
                        trans_dist[perp_state] = trans_dist.get(perp_state, 0) + self.noise / 2
                    else:
                        trans_dist[state] = trans_dist.get(state, 0) + self.noise / 2

                transitions[(state, action)] = trans_dist

                # Rewards
                if State(intended_next) == State(self.goal_pos):
                    rewards[(state, action)] = 1.0
                elif State(intended_next) == State(self.danger_pos):
                    rewards[(state, action)] = -1.0
                else:
                    rewards[(state, action)] = self.living_reward

        # Terminal rewards for display
        terminal_rewards = {
            State(self.goal_pos): 1.0,
            State(self.danger_pos): -1.0
        }

        return MDP(
            states=states,
            actions=self.actions,
            transitions=transitions,
            rewards=rewards,
            discount=self.discount,
            start_state=start_state,
            terminal_states=terminal_states,
            terminal_rewards=terminal_rewards
        )

    def _get_next_position(self, pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Get next position after taking action (respecting boundaries)"""
        dr, dc = self.action_effects[action]
        new_r = max(0, min(self.grid_size - 1, pos[0] + dr))
        new_c = max(0, min(self.grid_size - 1, pos[1] + dc))

        # Check if hit obstacle
        if (new_r, new_c) in self.obstacles:
            return pos  # Stay in place

        return (new_r, new_c)

    def _get_perpendicular_actions(self, action: str) -> List[str]:
        """Get perpendicular actions for noise"""
        if action in ["UP", "DOWN"]:
            return ["LEFT", "RIGHT"]
        else:
            return ["UP", "DOWN"]

    def get_mdp(self) -> MDP:
        """Get MDP"""
        return self.mdp
