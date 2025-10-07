"""
Maze environment for search algorithms
"""

import random
from typing import List, Tuple, Set
from modules.search.core.state import State

class Maze:
    """Maze environment with procedural generation"""

    def __init__(
        self,
        width: int,
        height: int,
        complexity: float = 0.75,
        start_pos: Tuple[int, int] = None,
        goal_pos: Tuple[int, int] = None,
        random_start_goal: bool = False
    ):
        self.width = width
        self.height = height
        self.complexity = complexity

        # Initialize grid (0 = passable, 1 = wall)
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

        # Set start and goal positions
        if random_start_goal:
            # Random positions
            self.start = self._get_random_position()
            self.goal = self._get_random_position(exclude=[self.start])
        else:
            # Use provided or default positions
            self.start = start_pos if start_pos else (1, 1)
            self.goal = goal_pos if goal_pos else (height - 2, width - 2)

        self.generate_maze()

    def generate_maze(self):
        """Generate maze using recursive backtracking"""
        # Initialize with walls
        for i in range(self.height):
            for j in range(self.width):
                if i == 0 or i == self.height - 1 or j == 0 or j == self.width - 1:
                    self.grid[i][j] = 1  # Border walls

        # Carve passages using DFS
        stack = [self.start]
        visited = {self.start}
        self.grid[self.start[0]][self.start[1]] = 0

        while stack:
            current = stack[-1]

            # Get unvisited neighbors (2 cells away)
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = current[0] + dr, current[1] + dc
                if (0 < nr < self.height - 1 and
                    0 < nc < self.width - 1 and
                    (nr, nc) not in visited):
                    neighbors.append((nr, nc))

            if neighbors and random.random() < self.complexity:
                # Choose random neighbor
                next_cell = random.choice(neighbors)

                # Remove wall between current and next
                wall_r = (current[0] + next_cell[0]) // 2
                wall_c = (current[1] + next_cell[1]) // 2
                self.grid[wall_r][wall_c] = 0
                self.grid[next_cell[0]][next_cell[1]] = 0

                # Mark as visited and add to stack
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()

        # Ensure start and goal are clear
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0

        # Add some random walls for variety
        self._add_random_obstacles()

    def _add_random_obstacles(self):
        """Add random obstacles for complexity"""
        num_obstacles = int(self.width * self.height * 0.05)
        for _ in range(num_obstacles):
            r = random.randint(2, self.height - 3)
            c = random.randint(2, self.width - 3)
            if (r, c) != self.start and (r, c) != self.goal:
                self.grid[r][c] = 1

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (in bounds and not a wall)"""
        r, c = pos
        return (0 <= r < self.height and
                0 <= c < self.width and
                self.grid[r][c] == 0)

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        r, c = pos
        neighbors = []

        # 4-connected (up, right, down, left)
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            new_pos = (r + dr, c + dc)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)

        return neighbors

    def get_action_name(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Get action name for moving from one position to another"""
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]

        if dr == -1: return "UP"
        if dr == 1: return "DOWN"
        if dc == 1: return "RIGHT"
        if dc == -1: return "LEFT"
        return "STAY"

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def _get_random_position(self, exclude: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Get random valid position in maze"""
        exclude = exclude or []
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            r = random.randint(1, self.height - 2)
            c = random.randint(1, self.width - 2)
            pos = (r, c)

            if pos not in exclude:
                return pos

            attempts += 1

        # Fallback to corners if random fails
        corners = [
            (1, 1),
            (1, self.width - 2),
            (self.height - 2, 1),
            (self.height - 2, self.width - 2)
        ]
        for corner in corners:
            if corner not in exclude:
                return corner

        return (1, 1)  # Final fallback

    def set_start_goal(self, start: Tuple[int, int] = None, goal: Tuple[int, int] = None):
        """Set start and goal positions after maze creation"""
        if start:
            self.start = start
            self.grid[start[0]][start[1]] = 0  # Ensure passable

        if goal:
            self.goal = goal
            self.grid[goal[0]][goal[1]] = 0  # Ensure passable
