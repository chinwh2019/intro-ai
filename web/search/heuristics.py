"""
Heuristic functions for informed search algorithms

Students can create custom heuristics here and use them with A* or Greedy search.

Example usage:
    from heuristics import manhattan_distance, my_custom_heuristic
    from algorithms.astar import AStar

    # Use built-in heuristic
    astar = AStar(maze, heuristic_func=manhattan_distance)

    # Use custom heuristic
    astar = AStar(maze, heuristic_func=my_custom_heuristic)
"""

from typing import Tuple, Callable
import math


# Built-in heuristics

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Manhattan distance (L1 norm)
    Admissible for 4-connected grid movement

    Args:
        pos1: Starting position (row, col)
        pos2: Goal position (row, col)

    Returns:
        Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Euclidean distance (L2 norm)
    Admissible but may underestimate for grid movement

    Args:
        pos1: Starting position (row, col)
        pos2: Goal position (row, col)

    Returns:
        Euclidean distance
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def chebyshev_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Chebyshev distance (L∞ norm)
    Admissible for 8-connected movement (with diagonals)

    Args:
        pos1: Starting position (row, col)
        pos2: Goal position (row, col)

    Returns:
        Chebyshev distance
    """
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def zero_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Zero heuristic (always returns 0)
    Makes A* behave like UCS
    Admissible but not informative

    Args:
        pos1: Starting position (row, col)
        pos2: Goal position (row, col)

    Returns:
        0
    """
    return 0.0


# Custom heuristics - Students can add their own here!

def weighted_manhattan(pos1: Tuple[int, int], pos2: Tuple[int, int], weight: float = 1.0) -> float:
    """
    Weighted Manhattan distance
    weight > 1.0: Inadmissible but may find solution faster
    weight < 1.0: Still admissible but less informed

    Args:
        pos1: Starting position
        pos2: Goal position
        weight: Multiplier for heuristic value

    Returns:
        Weighted Manhattan distance
    """
    return weight * manhattan_distance(pos1, pos2)


def diagonal_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Diagonal distance for 8-connected grids
    Assumes diagonal moves cost sqrt(2) and straight moves cost 1

    Args:
        pos1: Starting position
        pos2: Goal position

    Returns:
        Diagonal distance
    """
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])

    # Diagonal moves
    diagonal_moves = min(dx, dy)
    straight_moves = abs(dx - dy)

    return math.sqrt(2) * diagonal_moves + straight_moves


# Example custom heuristic for students to modify

def my_custom_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    STUDENT ACTIVITY: Create your own heuristic!

    Try different ideas:
    - Combine multiple distance metrics
    - Add penalties or bonuses
    - Use domain knowledge

    Remember: For A* to guarantee optimality, heuristic must be admissible
    (never overestimate the true cost to goal)

    Args:
        pos1: Current position (row, col)
        pos2: Goal position (row, col)

    Returns:
        Estimated cost to goal
    """
    # Example: weighted combination of Manhattan and Euclidean
    manhattan = manhattan_distance(pos1, pos2)
    euclidean = euclidean_distance(pos1, pos2)

    # You can modify this!
    return 0.7 * manhattan + 0.3 * euclidean


# Heuristic factory

def create_weighted_heuristic(weight: float = 1.0) -> Callable:
    """
    Create a weighted Manhattan heuristic with custom weight

    Args:
        weight: Heuristic weight (>1 = inadmissible, faster; <1 = more admissible)

    Returns:
        Heuristic function
    """
    def heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return weight * manhattan_distance(pos1, pos2)

    return heuristic


def create_combined_heuristic(alpha: float = 0.5) -> Callable:
    """
    Create heuristic that combines Manhattan and Euclidean

    Args:
        alpha: Weight for Manhattan (1-alpha for Euclidean)

    Returns:
        Heuristic function
    """
    def heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        manhattan = manhattan_distance(pos1, pos2)
        euclidean = euclidean_distance(pos1, pos2)
        return alpha * manhattan + (1 - alpha) * euclidean

    return heuristic


# Heuristic registry for easy access

HEURISTICS = {
    'manhattan': manhattan_distance,
    'euclidean': euclidean_distance,
    'chebyshev': chebyshev_distance,
    'zero': zero_heuristic,
    'diagonal': diagonal_distance,
    'custom': my_custom_heuristic,
}


def get_heuristic(name: str) -> Callable:
    """
    Get heuristic function by name

    Args:
        name: Heuristic name (see HEURISTICS dict)

    Returns:
        Heuristic function
    """
    if name in HEURISTICS:
        return HEURISTICS[name]
    else:
        print(f"Unknown heuristic: {name}")
        print(f"Available: {list(HEURISTICS.keys())}")
        return manhattan_distance  # Default
