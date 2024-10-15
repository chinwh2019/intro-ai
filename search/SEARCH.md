# Maze Solver: DFS vs BFS

This Python program implements a maze solver using Pygame, demonstrating the differences between Depth-First Search (DFS) and Breadth-First Search (BFS) algorithms in pathfinding.

## Overview

The program generates a random maze and allows the user to visualize how DFS and BFS algorithms traverse the maze to find a path from the start point to the end point. The maze is represented as a grid where white cells are passable, and black cells are walls.

## How It Works

### Depth-First Search (DFS)

DFS explores the maze by going as deep as possible along each branch before backtracking:

1. Start at the initial position (start point).
2. Explore an adjacent unvisited cell.
3. Continue this process, always choosing an unvisited neighbor when possible.
4. If there are no unvisited neighbors, backtrack to the previous cell.
5. Repeat steps 2-4 until the end point is reached or all possible paths are explored.

Implementation details:
- Uses a stack to keep track of the cells to visit.
- Explores in the order: right, down, left, up.
- Yields the current state (visited cells and path) at each step for visualization.

### Breadth-First Search (BFS)

BFS explores the maze by visiting all neighbors at the present depth before moving to nodes at the next depth level:

1. Start at the initial position (start point).
2. Explore all adjacent cells.
3. For each of these adjacent cells, explore their adjacent cells that haven't been visited.
4. Repeat step 3 until the end point is reached or all cells are visited.

Implementation details:
- Uses a queue to keep track of the cells to visit.
- Explores in the order: right, down, left, up.
- Yields the current state (visited cells and path) at each step for visualization.


### Uniform Cost Search (UCS)

UCS explores the maze by always expanding the node with the lowest path cost:

1. Start at the initial position (start point) with a cost of 0.
2. Maintain a priority queue of nodes to visit, ordered by their path cost.
3. Always expand the node with the lowest cost from the priority queue.
4. For each neighbor of the expanded node, calculate its path cost and add it to the queue if it hasn't been visited.
5. Repeat steps 3-4 until the end point is reached or all possible paths are explored.

Implementation details:
- Uses a priority queue (implemented with a heap) to keep track of the cells to visit.
- Each cell in the queue is associated with its total path cost.
- Explores in the order: right, down, left, up.
- Yields the current state (visited cells and path) at each step for visualization.

### A* Search

A* is an informed search algorithm that combines the benefits of UCS and a heuristic function:

1. Start at the initial position (start point).
2. Maintain a priority queue of nodes to visit, ordered by f(n) = g(n) + h(n), where:
   - g(n) is the actual cost from the start to the current node
   - h(n) is the estimated cost from the current node to the goal (heuristic)
3. Always expand the node with the lowest f(n) from the priority queue.
4. For each neighbor of the expanded node, calculate its f(n) and add it to the queue if it hasn't been visited or if a better path is found.
5. Repeat steps 3-4 until the end point is reached or all possible paths are explored.

Implementation details:
- Uses a priority queue (implemented with a heap) to keep track of the cells to visit.
- Uses Manhattan distance as the heuristic function.
- Explores in the order: right, down, left, up.
- Yields the current state (visited cells and path) at each step for visualization.

## Visualization

- Green cell: Start point
- Red cell: End point
- Yellow cells: Visited nodes
- Blue cells: Current path

The program provides real-time visualization of the algorithm's progress, allowing users to see how DFS and BFS differ in their exploration patterns.

## Controls

- Press 'D' to start Depth-First Search (DFS)
- Press 'B' to start Breadth-First Search (BFS)
- Press 'R' to regenerate the maze

## Performance

The program displays the total time taken to find the solution, allowing users to compare the performance of DFS and BFS for different maze configurations.