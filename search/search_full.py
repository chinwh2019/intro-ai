import heapq
import random
import time
from collections import deque

import pygame

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
GUIDELINE_HEIGHT = 150

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT + GUIDELINE_HEIGHT))
pygame.display.set_caption("Maze Solver: DFS vs BFS vs UCS vs A*")


class Maze:
    def __init__(self):
        self.grid = [[1 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.start = (1, 1)
        self.end = (
            GRID_HEIGHT - 3,
            GRID_WIDTH - 3,
        )  # Ensure start and end are always within bounds
        self.generate_maze()

    def generate_maze(self):
        # Improved maze generation using DFS with backtracking
        stack = [self.start]
        self.grid[self.start[0]][self.start[1]] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []

            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if (
                    1 <= nx < GRID_HEIGHT - 1
                    and 1 <= ny < GRID_WIDTH - 1
                    and self.grid[nx][ny] == 1
                ):
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = random.choice(neighbors)
                stack.append((nx, ny))
                self.grid[nx][ny] = 0
                self.grid[(nx + x) // 2][(ny + y) // 2] = 0
            else:
                stack.pop()

    def draw(self):
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                color = BLACK if self.grid[i][j] == 1 else WHITE
                pygame.draw.rect(
                    screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

        pygame.draw.rect(
            screen,
            GREEN,
            (
                self.start[1] * CELL_SIZE,
                self.start[0] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            ),
        )
        pygame.draw.rect(
            screen,
            RED,
            (self.end[1] * CELL_SIZE, self.end[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )


def dfs(maze, start, end):
    stack = [(start, [start])]
    visited = set()
    nodes_expanded = 0

    while stack:
        (x, y), path = stack.pop()
        nodes_expanded += 1
        if (x, y) == end:
            return path, nodes_expanded

        if (x, y) not in visited:
            visited.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < GRID_HEIGHT
                    and 0 <= ny < GRID_WIDTH
                    and maze.grid[nx][ny] == 0
                ):
                    stack.append(((nx, ny), path + [(nx, ny)]))

        yield visited, path

    return None, nodes_expanded


def bfs(maze, start, end):
    queue = deque([(start, [start])])
    visited = set()
    nodes_expanded = 0

    while queue:
        (x, y), path = queue.popleft()
        nodes_expanded += 1
        if (x, y) == end:
            return path, nodes_expanded

        if (x, y) not in visited:
            visited.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < GRID_HEIGHT
                    and 0 <= ny < GRID_WIDTH
                    and maze.grid[nx][ny] == 0
                ):
                    queue.append(((nx, ny), path + [(nx, ny)]))

        yield visited, path

    return None, nodes_expanded


def ucs(maze, start, end):
    pq = [(0, start, [start])]
    visited = set()
    nodes_expanded = 0

    while pq:
        cost, (x, y), path = heapq.heappop(pq)
        nodes_expanded += 1
        if (x, y) == end:
            return path, nodes_expanded

        if (x, y) not in visited:
            visited.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < GRID_HEIGHT
                    and 0 <= ny < GRID_WIDTH
                    and maze.grid[nx][ny] == 0
                ):
                    heapq.heappush(pq, (cost + 1, (nx, ny), path + [(nx, ny)]))

        yield visited, path

    return None, nodes_expanded


def astar(maze, start, end):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    pq = [(heuristic(start, end), 0, start, [start])]
    visited = set()
    nodes_expanded = 0

    while pq:
        _, cost, (x, y), path = heapq.heappop(pq)
        nodes_expanded += 1
        if (x, y) == end:
            return path, nodes_expanded

        if (x, y) not in visited:
            visited.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < GRID_HEIGHT
                    and 0 <= ny < GRID_WIDTH
                    and maze.grid[nx][ny] == 0
                ):
                    heapq.heappush(
                        pq,
                        (
                            cost + 1 + heuristic((nx, ny), end),
                            cost + 1,
                            (nx, ny),
                            path + [(nx, ny)],
                        ),
                    )

        yield visited, path

    return None, nodes_expanded


def main():
    maze = Maze()
    clock = pygame.time.Clock()
    running = True
    solving = False
    algorithm = None
    path_generator = None
    final_path = None
    start_time = None
    total_time = None
    nodes_expanded = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    solving = True
                    algorithm = "DFS"
                    path_generator = dfs(maze, maze.start, maze.end)
                    final_path = None
                    start_time = time.time()
                elif event.key == pygame.K_b:
                    solving = True
                    algorithm = "BFS"
                    path_generator = bfs(maze, maze.start, maze.end)
                    final_path = None
                    start_time = time.time()
                elif event.key == pygame.K_u:
                    solving = True
                    algorithm = "UCS"
                    path_generator = ucs(maze, maze.start, maze.end)
                    final_path = None
                    start_time = time.time()
                elif event.key == pygame.K_a:
                    solving = True
                    algorithm = "A*"
                    path_generator = astar(maze, maze.start, maze.end)
                    final_path = None
                    start_time = time.time()
                elif event.key == pygame.K_r:
                    maze = Maze()
                    solving = False
                    final_path = None
                    total_time = None
                    nodes_expanded = 0

        screen.fill(WHITE)
        maze.draw()

        if solving:
            try:
                visited, current_path = next(path_generator)
                for x, y in visited:
                    pygame.draw.rect(
                        screen,
                        YELLOW,
                        (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    )
                for x, y in current_path:
                    pygame.draw.rect(
                        screen,
                        BLUE,
                        (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    )
            except StopIteration as e:
                solving = False
                result = e.value
                if result:
                    final_path, nodes_expanded = result
                else:
                    final_path = current_path
                if start_time:
                    total_time = time.time() - start_time

        if final_path:
            for x, y in final_path:
                pygame.draw.rect(
                    screen, BLUE, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

        # Draw the guidelines for the user outside the maze area
        font = pygame.font.Font(None, 24)
        guidelines = [
            "Press 'D' to start Depth-First Search (DFS)",
            "Press 'B' to start Breadth-First Search (BFS)",
            "Press 'U' to start Uniform Cost Search (UCS)",
            "Press 'A' to start A* Search",
            "Press 'R' to regenerate the maze",
            "Green cell: Start point",
            "Red cell: End point",
            "Yellow cells: Visited nodes",
            "Blue cells: Current path",
        ]
        for idx, line in enumerate(guidelines):
            text = font.render(line, True, BLACK)
            screen.blit(text, (10, HEIGHT + 10 + idx * 20))

        # Display the total time taken if the solution is found
        if total_time is not None:
            time_text = font.render(
                f"Total time taken: {total_time:.2f} seconds", True, BLACK
            )
            screen.blit(time_text, (WIDTH - 300, HEIGHT + 10))

        # Display the number of nodes expanded if the solution is found
        if nodes_expanded > 0:
            nodes_text = font.render(f"Nodes expanded: {nodes_expanded}", True, BLACK)
            screen.blit(nodes_text, (WIDTH - 300, HEIGHT + 40))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
