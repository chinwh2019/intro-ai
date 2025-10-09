# Search Algorithms Module

Interactive visualization of classic search algorithms for pathfinding in mazes.

## Features

- **5 Search Algorithms**: BFS, DFS, UCS, A*, Greedy Best-First
- **Real-time Visualization**: Watch algorithms explore the state space step-by-step
- **Performance Metrics**: Compare nodes expanded, path length, and efficiency
- **Interactive Parameter Panel**: Adjust speed, heuristic weight, and complexity with sliders
- **Random Start/Goal Mode**: Toggle randomization for varied testing
- **Live Algorithm Switching**: Change algorithms on-the-fly with number keys
- **Step-by-step Control**: Pause and step through search for detailed analysis

## Quick Start

```bash
# Install dependencies
pip install pygame

# Run the module
python scripts/run_search.py

# Or use a preset
python scripts/run_search.py --preset fast
python scripts/run_search.py --preset simple
```

## Controls

### Keyboard Shortcuts
- **1-5**: Select algorithm (1=BFS, 2=DFS, 3=UCS, 4=A*, 5=Greedy)
- **SPACE**: Pause/Resume search
- **S**: Step through one iteration (when paused)
- **R**: Reset and generate new maze
- **T**: Toggle random start/goal mode
- **Q**: Quit application

### Interactive UI Panel (Bottom Left)
- **Speed Slider**: Adjust animation speed (0.1x to 10x)
- **Heuristic Weight Slider**: Control A* and Greedy heuristic strength (0.5 to 3.0)
- **Complexity Slider**: Set maze complexity for next reset (0.3 to 1.0)
- **Apply Button**: Apply slider changes immediately

**Features:**
- Sliders update in real-time when you click Apply
- Heuristic weight > 1.0 makes A*/Greedy faster but potentially suboptimal (inadmissible)
- Changing complexity takes effect on next maze reset (press R)
- Speed changes apply immediately to current search

## Algorithms Implemented

### 1. Breadth-First Search (BFS)
- **Strategy**: Explore level by level
- **Completeness**: Yes
- **Optimality**: Yes (for unweighted graphs)
- **Space**: O(b^d) - exponential

### 2. Depth-First Search (DFS)
- **Strategy**: Explore deeply before backtracking
- **Completeness**: Yes (in finite spaces)
- **Optimality**: No
- **Space**: O(bd) - linear

### 3. Uniform Cost Search (UCS)
- **Strategy**: Expand least-cost node first
- **Completeness**: Yes
- **Optimality**: Yes
- **Space**: O(b^d)

### 4. A* Search
- **Strategy**: UCS + heuristic guidance
- **Completeness**: Yes
- **Optimality**: Yes (with admissible heuristic)
- **Space**: O(b^d) but typically much better
- **Heuristic**: Manhattan distance (default)

### 5. Greedy Best-First Search
- **Strategy**: Expand node closest to goal (by heuristic)
- **Completeness**: No
- **Optimality**: No
- **Space**: O(b^m) where m is max depth

## Configuration

### Method 1: Use Presets (Easiest)

```python
from modules.search.config import load_preset

load_preset('fast')       # Fast animation
load_preset('simple')     # Small 20x15 maze
load_preset('detailed')   # Larger cells, show numbers
load_preset('large_maze') # 60x40 maze
```

### Method 2: Modify Parameters

```python
from modules.search.config import config

# Change maze size
config.MAZE_WIDTH = 50
config.MAZE_HEIGHT = 40

# Change animation
config.ANIMATION_SPEED = 2.0
config.STEP_DELAY = 0.02

# Change appearance
config.SHOW_NUMBERS = True
config.CELL_SIZE = 25
```

### Method 3: Custom Start/Goal Positions

```python
from modules.search.config import config

# Option A: Set specific positions
config.START_POSITION = (5, 5)    # (row, col)
config.GOAL_POSITION = (20, 35)   # (row, col)

# Option B: Randomize on each reset
config.RANDOM_START_GOAL = True
```

### Method 4: Custom Heuristics for A*

```python
from modules.search.algorithms.astar import AStar
from modules.search.core.environment import Maze

maze = Maze(30, 20)

# Use built-in heuristics
astar = AStar(maze, heuristic='manhattan')  # Default
astar = AStar(maze, heuristic='euclidean')  # Euclidean distance
astar = AStar(maze, heuristic='zero')       # No heuristic (= UCS)
astar = AStar(maze, heuristic='custom')     # Use my_custom_heuristic

# Or define your own heuristic function
def my_heuristic(pos1, pos2):
    manhattan = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    return 1.2 * manhattan  # Weighted (experiment!)

astar = AStar(maze, heuristic_func=my_heuristic)
```

See `modules/search/heuristics.py` for more heuristic examples and templates!

## Understanding the Visualization

### Layout

**Left Sidebar (300px):**
- Algorithm name and status
- Key mappings reference (1-5 for algorithms)
- Color-coded legend
- Performance statistics
- Random mode indicator
- Interactive parameter panel (sliders)

**Main Area:**
- Top control panel with instructions
- Maze visualization
  - **Green cell**: Start position
  - **Red cell**: Goal position
  - **Yellow cells**: Explored states
  - **Purple cells**: Frontier (to be explored)
  - **Cyan cells**: Solution path
  - **Orange cell**: Currently expanding
  - **Gray cells**: Walls

### Statistics Explained

- **Nodes Expanded**: How many states the algorithm examined
- **Nodes Generated**: How many states were created (including duplicates)
- **Max Frontier**: Largest size the frontier reached (space complexity)
- **Steps**: Total search iterations
- **Path Length**: Number of moves in solution

**Lower is better** for all metrics except path length (which should be optimal for BFS, UCS, A*).

## Web Version

A browser-compatible version is available at `web/search/`:
- No installation required for students
- Runs in any modern browser
- Same features as desktop (with interactive controls!)
- Deployed via GitHub Pages for class-wide access

## File Structure

```
modules/search/
├── __init__.py
├── README.md (this file)
├── config.py                 # Configuration settings
├── core/
│   ├── __init__.py
│   ├── state.py             # State and Node classes
│   ├── environment.py       # Maze generation
│   └── base_algorithm.py    # Base class for algorithms
├── algorithms/
│   ├── __init__.py
│   ├── bfs.py               # Breadth-First Search
│   ├── dfs.py               # Depth-First Search
│   ├── ucs.py               # Uniform Cost Search
│   ├── astar.py             # A* Search
│   └── greedy.py            # Greedy Best-First
├── ui/
│   ├── __init__.py
│   └── visualizer.py        # Visualization system
└── main.py                  # Main application
```

## For Students

### Beginner Activities
1. Run each algorithm and observe behavior differences
2. Compare performance metrics
3. Modify colors and animation speed
4. Try different maze sizes

### Intermediate Activities
1. Experiment with different heuristics in A*
2. Create custom maze complexity settings
3. Benchmark algorithm performance
4. Analyze when each algorithm is best

### Advanced Activities
1. Implement new algorithms (bidirectional search, IDA*)
2. Create custom heuristic functions
3. Add diagonal movement support
4. Implement maze editing interface

## Learning Objectives

Students will:
- Understand state-space search
- Recognize trade-offs: completeness, optimality, time, space
- Develop intuition for algorithm selection
- Learn heuristic design for A*
- Compare uninformed vs informed search

## Technical Details

- **Language**: Python 3.8+
- **Graphics**: Pygame
- **Architecture**: Generator-based for step-by-step execution
- **State Representation**: Immutable State objects
- **Search Tree**: Node objects with parent pointers

## License

MIT License - See root LICENSE file
