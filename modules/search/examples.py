"""
Student Examples for Search Module

This file demonstrates how to:
1. Load presets
2. Modify configuration
3. Change start/end points
4. Use custom heuristics
5. Run and compare algorithms

Run any example function to see it in action!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Example 1: Using Presets
def example_use_presets():
    """Example: Load different presets"""
    from modules.search.config import config, load_preset
    from modules.search.core.environment import Maze

    print("Example 1: Using Presets")
    print("=" * 50)

    # Try different presets
    presets_to_try = ['simple', 'fast', 'detailed']

    for preset_name in presets_to_try:
        load_preset(preset_name)

        print(f"\nPreset: {preset_name}")
        print(f"  Maze size: {config.MAZE_WIDTH}x{config.MAZE_HEIGHT}")
        print(f"  Animation speed: {config.ANIMATION_SPEED}x")
        print(f"  Cell size: {config.CELL_SIZE}px")

        # Create maze with this preset
        maze = Maze(config.MAZE_WIDTH, config.MAZE_HEIGHT, config.MAZE_COMPLEXITY)
        print(f"  Maze created: {maze.width}x{maze.height}")


# Example 2: Custom Configuration
def example_custom_config():
    """Example: Customize parameters"""
    from modules.search.config import config
    from modules.search.core.environment import Maze

    print("\nExample 2: Custom Configuration")
    print("=" * 50)

    # Modify config
    config.MAZE_WIDTH = 25
    config.MAZE_HEIGHT = 20
    config.MAZE_COMPLEXITY = 0.5  # Easier maze
    config.ANIMATION_SPEED = 3.0   # Faster animation

    print(f"Custom settings:")
    print(f"  Maze size: {config.MAZE_WIDTH}x{config.MAZE_HEIGHT}")
    print(f"  Complexity: {config.MAZE_COMPLEXITY}")

    maze = Maze(config.MAZE_WIDTH, config.MAZE_HEIGHT, config.MAZE_COMPLEXITY)
    print(f"✓ Maze created with custom settings")


# Example 3: Custom Start and Goal Positions
def example_custom_start_goal():
    """Example: Set specific start and goal positions"""
    from modules.search.config import config
    from modules.search.core.environment import Maze

    print("\nExample 3: Custom Start and Goal")
    print("=" * 50)

    # Option A: Set specific positions
    config.START_POSITION = (5, 5)
    config.GOAL_POSITION = (15, 25)

    maze = Maze(
        width=30,
        height=20,
        complexity=0.6,
        start_pos=config.START_POSITION,
        goal_pos=config.GOAL_POSITION
    )

    print(f"Specific positions:")
    print(f"  Start: {maze.start}")
    print(f"  Goal: {maze.goal}")

    # Option B: Random positions
    config.RANDOM_START_GOAL = True

    maze2 = Maze(
        width=30,
        height=20,
        complexity=0.6,
        random_start_goal=True
    )

    print(f"\nRandom positions:")
    print(f"  Start: {maze2.start}")
    print(f"  Goal: {maze2.goal}")

    # Generate another maze - positions will be different
    maze3 = Maze(30, 20, 0.6, random_start_goal=True)
    print(f"\nAnother random:")
    print(f"  Start: {maze3.start}")
    print(f"  Goal: {maze3.goal}")


# Example 4: Using Custom Heuristics
def example_custom_heuristics():
    """Example: Use different heuristics with A*"""
    from modules.search.core.environment import Maze
    from modules.search.algorithms.astar import AStar
    from modules.search import heuristics

    print("\nExample 4: Custom Heuristics")
    print("=" * 50)

    # Create maze
    maze = Maze(25, 20)

    # Method 1: Use built-in heuristics by name
    algorithms = [
        ('Manhattan', AStar(maze, heuristic='manhattan')),
        ('Euclidean', AStar(maze, heuristic='euclidean')),
        ('Zero (UCS)', AStar(maze, heuristic='zero')),
        ('Custom', AStar(maze, heuristic='custom')),
    ]

    print("\nComparing different heuristics:")
    for name, algo in algorithms:
        # Run to completion
        for _ in algo.search():
            pass

        stats = algo.get_statistics()
        print(f"\n{name}:")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
        print(f"  Path length: {stats['solution_length']}")
        print(f"  Optimal: {stats['solution_length'] == algorithms[0][1].get_statistics()['solution_length']}")

    # Method 2: Pass heuristic function directly
    print("\n\nUsing heuristic function directly:")

    def my_heuristic(pos1, pos2):
        """My custom heuristic - weighted Manhattan"""
        manhattan = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        return 1.5 * manhattan  # Weight > 1 = inadmissible but faster

    astar_custom = AStar(maze, heuristic_func=my_heuristic)

    for _ in astar_custom.search():
        pass

    print(f"Custom heuristic:")
    print(f"  Nodes expanded: {astar_custom.nodes_expanded}")
    print(f"  Path length: {len(astar_custom.solution_path)}")


# Example 5: Creating Your Own Heuristic
def example_create_heuristic():
    """Example: Create and test your own heuristic"""
    from modules.search.core.environment import Maze
    from modules.search.algorithms.astar import AStar

    print("\nExample 5: Create Your Own Heuristic")
    print("=" * 50)

    # Step 1: Define your heuristic function
    def student_heuristic(pos1, pos2):
        """
        Student activity: Modify this function!

        Try:
        - Different distance metrics
        - Combinations of metrics
        - Adding weights or penalties
        """
        # Example: Combined heuristic
        manhattan = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        euclidean = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

        # Combine them (experiment with different weights!)
        return 0.6 * manhattan + 0.4 * euclidean

    # Step 2: Test it
    maze = Maze(30, 25)

    astar_student = AStar(maze, heuristic_func=student_heuristic)

    for _ in astar_student.search():
        pass

    print(f"Your heuristic:")
    print(f"  Nodes expanded: {astar_student.nodes_expanded}")
    print(f"  Solution found: {astar_student.solution_found}")
    print(f"  Path length: {len(astar_student.solution_path)}")

    # Step 3: Compare with standard heuristics
    astar_manhattan = AStar(maze, heuristic='manhattan')
    for _ in astar_manhattan.search():
        pass

    print(f"\nManhattan heuristic (for comparison):")
    print(f"  Nodes expanded: {astar_manhattan.nodes_expanded}")

    efficiency_gain = (
        (astar_manhattan.nodes_expanded - astar_student.nodes_expanded)
        / astar_manhattan.nodes_expanded * 100
    )
    print(f"\nYour heuristic is {efficiency_gain:.1f}% {'more' if efficiency_gain > 0 else 'less'} efficient")


# Example 6: Compare All Algorithms
def example_compare_all():
    """Example: Compare all algorithms on same maze"""
    from modules.search.core.environment import Maze
    from modules.search.algorithms.bfs import BFS
    from modules.search.algorithms.dfs import DFS
    from modules.search.algorithms.ucs import UCS
    from modules.search.algorithms.astar import AStar
    from modules.search.algorithms.greedy import GreedyBestFirst

    print("\nExample 6: Compare All Algorithms")
    print("=" * 50)

    # Create one maze
    maze = Maze(35, 25, complexity=0.7)
    print(f"Maze: {maze.width}x{maze.height}")
    print(f"Start: {maze.start}, Goal: {maze.goal}\n")

    # Create algorithms
    algorithms = [
        ('BFS', BFS(maze)),
        ('DFS', DFS(maze)),
        ('UCS', UCS(maze)),
        ('A* (Manhattan)', AStar(maze, heuristic='manhattan')),
        ('A* (Euclidean)', AStar(maze, heuristic='euclidean')),
        ('Greedy', GreedyBestFirst(maze)),
    ]

    # Run all algorithms
    results = []
    for name, algo in algorithms:
        for _ in algo.search():
            pass

        stats = algo.get_statistics()
        results.append({
            'name': name,
            'expanded': stats['nodes_expanded'],
            'generated': stats['nodes_generated'],
            'path_length': stats['solution_length'],
            'found': algo.solution_found,
        })

    # Print comparison table
    print(f"{'Algorithm':<20} {'Expanded':<10} {'Generated':<10} {'Path':<8} {'Optimal':<8}")
    print("-" * 70)

    optimal_length = min(r['path_length'] for r in results if r['found'])

    for r in results:
        optimal = "Yes" if r['path_length'] == optimal_length else "No"
        print(f"{r['name']:<20} {r['expanded']:<10} {r['generated']:<10} {r['path_length']:<8} {optimal:<8}")


# Example 7: Test Heuristic Admissibility
def example_test_admissibility():
    """Example: Test if a heuristic is admissible"""
    from modules.search.core.environment import Maze
    from modules.search.algorithms.ucs import UCS
    from modules.search.algorithms.astar import AStar

    print("\nExample 7: Test Heuristic Admissibility")
    print("=" * 50)

    maze = Maze(20, 15)

    # Get actual cost using UCS
    ucs = UCS(maze)
    for _ in ucs.search():
        pass
    actual_path_length = len(ucs.solution_path)

    # Test different heuristics
    heuristics_to_test = [
        ('Manhattan', 'manhattan'),
        ('Euclidean', 'euclidean'),
        ('Weighted 1.5x', lambda p1, p2: 1.5 * maze.manhattan_distance(p1, p2)),
        ('Weighted 0.5x', lambda p1, p2: 0.5 * maze.manhattan_distance(p1, p2)),
    ]

    print(f"Actual optimal path length (UCS): {actual_path_length}\n")

    for name, heuristic in heuristics_to_test:
        astar = AStar(maze, heuristic=heuristic)

        for _ in astar.search():
            pass

        path_length = len(astar.solution_path)
        is_optimal = path_length == actual_path_length

        admissible = "Admissible ✓" if is_optimal else "Inadmissible ✗"

        print(f"{name}:")
        print(f"  Path length: {path_length}")
        print(f"  {admissible}")
        print(f"  Nodes expanded: {astar.nodes_expanded}")
        print()


# Example 8: Run with All Features
def example_full_features():
    """Example: Use all features together"""
    print("\nExample 8: Full Features Demo")
    print("=" * 50)

    from modules.search.config import config, load_preset

    # Load preset
    load_preset('simple')

    # Override some settings
    config.RANDOM_START_GOAL = True  # Randomize positions
    config.ANIMATION_SPEED = 2.0     # Faster

    # Now run main app - start/goal will be random!
    print("\nConfiguration applied:")
    print(f"  Maze: {config.MAZE_WIDTH}x{config.MAZE_HEIGHT}")
    print(f"  Random start/goal: {config.RANDOM_START_GOAL}")
    print(f"  Speed: {config.ANIMATION_SPEED}x")

    print("\nRun: python run_search.py")
    print("Then press R to reset and see different start/goal positions!")


if __name__ == '__main__':
    print("Search Module Examples")
    print("=" * 70)
    print("\nChoose an example to run:")
    print("  1. Use presets")
    print("  2. Custom configuration")
    print("  3. Custom start and goal positions")
    print("  4. Custom heuristics")
    print("  5. Create your own heuristic")
    print("  6. Compare all algorithms")
    print("  7. Test heuristic admissibility")
    print("  8. Full features demo")
    print()

    choice = input("Enter number (1-8): ")

    examples = {
        '1': example_use_presets,
        '2': example_custom_config,
        '3': example_custom_start_goal,
        '4': example_custom_heuristics,
        '5': example_create_heuristic,
        '6': example_compare_all,
        '7': example_test_admissibility,
        '8': example_full_features,
    }

    if choice in examples:
        examples[choice]()
    else:
        print("Invalid choice!")
