# Breadth-First Search (BFS) - Tutorial

BFS is your first search algorithm - simple, complete, and optimal for unweighted graphs.

## The Big Idea

**Imagine you're in a corn maze:**
- You explore all paths of length 1
- Then all paths of length 2
- Then all paths of length 3
- Continue until you find the exit

**That's BFS** - explore layer by layer!

## How BFS Works

### Algorithm in Plain English

```
1. Start at the starting position
2. Add all neighbors to a "to-visit" list (queue)
3. Visit the first place in the list
4. Add ITS neighbors to the end of the list
5. Repeat steps 3-4 until you find the goal
```

**Key insight:** You always explore closer places before farther places.

### Step-by-Step Example

**Initial state:**
```
S . . .
. X . .
. . . G
```
(S=Start, G=Goal, X=Wall, .=Empty)

**Step 1:** Start at S, add neighbors
```
Frontier: [(1,0), (0,1)]
Explored: [S]
```

**Step 2:** Visit (1,0), add its neighbors
```
Frontier: [(0,1), (2,0), (1,1)]
Explored: [S, (1,0)]
```

**Step 3:** Visit (0,1), add its neighbors
```
Frontier: [(2,0), (1,1), (0,2)]
Explored: [S, (1,0), (0,1)]
```

**Continue...** until goal is reached!

## Watch It in Action

### Run BFS

```bash
python scripts/run_search.py
```

Press `1` to run BFS.

### What to Observe

**Yellow cells spreading:**
1. Start shows yellow neighbors first
2. Then neighbors of neighbors
3. Spreads outward in all directions equally
4. Forms concentric "rings" or "waves"

**Purple frontier:**
- Shows cells waiting to be explored
- Always grows outward from start
- New cells added to END of queue

**Cyan path:**
- Appears when goal is found
- Always the shortest path
- Traced back from goal to start

## The Code (Simplified)

```python
def bfs(maze, start, goal):
    # Queue: First In, First Out (FIFO)
    frontier = Queue()
    frontier.put(start)

    explored = set()
    parent = {}  # To reconstruct path

    while not frontier.empty():
        current = frontier.get()  # Get FIRST item

        if current == goal:
            return reconstruct_path(parent, goal)

        explored.add(current)

        for neighbor in get_neighbors(current):
            if neighbor not in explored and neighbor not in frontier:
                frontier.put(neighbor)  # Add to END
                parent[neighbor] = current

    return None  # No path found
```

**Key data structure:** **Queue** (FIFO - First In, First Out)

## Why BFS Finds the Shortest Path

**Proof by contradiction:**

1. Suppose BFS finds path of length k
2. Suppose shorter path of length k-1 exists
3. BFS explores ALL paths of length k-1 before ANY path of length k
4. Therefore, BFS would have found the shorter path first
5. Contradiction! BFS must find shortest path.

**Intuition:** BFS explores by distance from start. It MUST find all short paths before any long paths.

## Experiments

### Experiment 1: Watch the Waves

1. Run BFS on a maze
2. Pause it (SPACE) after ~20 steps
3. Notice yellow cells form a "ring" around start
4. Step through (S key) and watch ring expand
5. Resume (SPACE) and see it complete

**Question:** Why does it look like water spreading?

### Experiment 2: Compare with DFS

1. Run BFS, note "Nodes Expanded" and "Path Length"
2. Reset (R key)
3. Run DFS (press 2), note same statistics
4. Compare:
   - Which expanded more nodes?
   - Which found shorter path?
   - Which finished faster?

### Experiment 3: Maze Size Effect

```bash
# Small maze
python scripts/run_search.py --preset simple
# Press 1 for BFS, note nodes expanded

# Large maze
python scripts/run_search.py --preset large_maze
# Press 1 for BFS, note nodes expanded
```

**Question:** How does nodes expanded scale with maze size?

## Strengths and Weaknesses

### ✅ Strengths

1. **Complete**: Always finds solution if one exists
2. **Optimal**: Finds shortest path (unweighted)
3. **Simple**: Easy to understand and implement
4. **Systematic**: Explores every possibility

### ❌ Weaknesses

1. **Memory hungry**: Stores entire frontier (can be huge!)
2. **No goal guidance**: Explores in all directions equally
3. **Slow for large spaces**: Exponential time complexity
4. **Doesn't handle weights**: Assumes all edges cost 1

## When to Use BFS

**Use BFS when:**
- ✅ Graph is unweighted (all moves cost same)
- ✅ Need to guarantee shortest path
- ✅ State space is not too large
- ✅ You have enough memory

**Don't use BFS when:**
- ❌ Graph has weights (use UCS or A*)
- ❌ State space is huge (use A* or IDA*)
- ❌ Memory is very limited (use DFS or IDA*)
- ❌ Any solution is fine (use DFS)

## BFS in the Real World

### Social Networks
**"Six degrees of separation"**
- BFS finds shortest connection path
- Facebook friend suggestions
- LinkedIn connection paths

### Network Routing
**Finding shortest hop path**
- Router finds fewest hops to destination
- Ignores bandwidth (unweighted)

### Puzzle Solving
**Rubik's cube solver**
- BFS guarantees optimal solution
- State space is huge (43 quintillion states!)
- Need heuristics (A*) for practical solving

## Understanding the Frontier

### Queue Behavior (FIFO)

```python
# BFS uses a queue
frontier = Queue()

# Add items to BACK
frontier.put(A)  # Queue: [A]
frontier.put(B)  # Queue: [A, B]
frontier.put(C)  # Queue: [A, B, C]

# Remove from FRONT
item = frontier.get()  # Returns A, Queue: [B, C]
item = frontier.get()  # Returns B, Queue: [C]
```

**This ensures:** First added = First explored (layer-by-layer)

### Visualizing the Queue

**Initial:**
```
Frontier: [Start]
Explored: []
```

**After 1 expansion:**
```
Frontier: [N1, N2, N3, N4]  (Start's neighbors)
Explored: [Start]
```

**After 2 expansions:**
```
Frontier: [N2, N3, N4, N1a, N1b]  (N1's neighbors added to end)
Explored: [Start, N1]
```

**Notice:** Always process from front, add to back!

## Common Questions

**Q: Why is BFS so slow on large mazes?**
A: It explores EVERY cell at distance d before ANY cell at distance d+1. No shortcuts!

**Q: Can I make BFS faster?**
A: Not without changing the algorithm. For speed, use A* with a good heuristic.

**Q: Why does BFS explore so many cells?**
A: It has no information about goal location, so it explores uniformly in all directions.

**Q: Is BFS always optimal?**
A: Only for unweighted graphs! If edges have different costs, use UCS or A*.

**Q: What's the difference between frontier and explored?**
A: Frontier = know about but haven't expanded yet. Explored = already expanded.

## Try It Yourself

### Activity 1: Trace BFS by Hand

Draw this small maze on paper:
```
S . X
. X .
. . G
```

Trace BFS step-by-step:
1. Start frontier: [S]
2. Expand S, frontier: [(1,0), (0,1)]
3. Expand (1,0), frontier: [(0,1), (2,0)]
4. Continue...

**Check:** Did you find path S → (1,0) → (2,0) → (2,1) → (2,2) → G?

### Activity 2: Modify the Code

Open `modules/search/algorithms/bfs.py` and find the main loop.

**Challenge:** Add a print statement to show each state as it's expanded:
```python
print(f"Exploring: {current_node.state.position}")
```

Run it and watch the console - see the layer-by-layer exploration!

## Next Steps

- Learn [DFS (Depth-First Search)](DFS-Tutorial)
- Learn [A* (Informed search)](A-Star-Explained)
- Try [Search Exercises](Search-Exercises)
- Compare [All Algorithms](Algorithm-Comparison)

---

**Master BFS first - it's the foundation for understanding all other search algorithms!**
