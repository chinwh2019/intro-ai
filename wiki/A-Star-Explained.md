# A* Search - Explained

A* (pronounced "A-star") is the **gold standard** for pathfinding. It's optimal, efficient, and used everywhere from video games to GPS navigation.

## The Big Idea

**A* = UCS + Heuristic Guidance**

Think of it as combining:
- **What you know**: Actual cost to reach current state (like UCS)
- **What you estimate**: Estimated cost to reach goal (heuristic)

**Result:** Finds optimal path while exploring fewer nodes than BFS!

## The Core Equation

```
f(n) = g(n) + h(n)
```

Where:
- **f(n)**: Total estimated cost through node n
- **g(n)**: Actual cost from start to n (known)
- **h(n)**: Estimated cost from n to goal (heuristic)

**A* always expands the node with lowest f(n).**

## Visual Intuition

### What Each Component Represents

**g(n) - Backwards Looking:**
```
Start → → → → n
[4 steps taken]
g(n) = 4
```

**h(n) - Forward Looking:**
```
n → → → Goal
[estimated 3 steps]
h(n) = 3
```

**f(n) - Total Estimate:**
```
Start → → → → n → → → Goal
[4 actual]   [3 estimated]
f(n) = 4 + 3 = 7
```

**A* chooses:** The path with smallest total estimated cost!

## Step-by-Step Example

### Simple 4×4 Maze

```
S . . .
. X X .
. . . .
. X X G
```

### Iteration 1: Expand Start

**Current:** S at (0,0)
- g=0, h=6, f=6

**Neighbors:**
- (1,0): g=1, h=5, f=6
- (0,1): g=1, h=5, f=6

**Frontier:** [(1,0,f=6), (0,1,f=6)]

### Iteration 2: Expand (1,0)

**Current:** (1,0)
- g=1, h=5, f=6

**Neighbors:**
- (2,0): g=2, h=4, f=6
- (1,1): WALL - skip

**Frontier:** [(0,1,f=6), (2,0,f=6)]

### Continue Until Goal...

**Final Path:** S → (1,0) → (2,0) → (2,1) → (2,2) → (3,2) → (3,3) → G

**Total Cost:** 6 steps (optimal!)

## Watch It in Action

### Run A*

```bash
python scripts/run_search.py
```

Press `4` to run A*.

### What to Observe

1. **Exploration bias toward goal:**
   - Yellow cells concentrate in goal direction
   - Forms elongated shape, not circle (like BFS)

2. **Fewer nodes explored:**
   - Compare "Nodes Expanded" with BFS
   - A* typically 30-50% fewer

3. **Same path length:**
   - Path length = BFS path length
   - Optimal solution guaranteed!

4. **Purple frontier:**
   - Cells closest to goal (by f-value)
   - Not uniform like BFS

## The Code (Simplified)

```python
def astar(maze, start, goal, heuristic):
    # Priority queue: lowest f(n) first
    frontier = PriorityQueue()
    frontier.put((0, start))  # (f_value, node)

    g_cost = {start: 0}
    explored = set()

    while not frontier.empty():
        f, current = frontier.get()

        if current == goal:
            return reconstruct_path(current)

        explored.add(current)

        for neighbor in get_neighbors(current):
            new_g = g_cost[current] + 1  # Cost to reach neighbor

            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                h = heuristic(neighbor, goal)
                f = new_g + h
                frontier.put((f, neighbor))

    return None
```

**Key data structure:** **Priority Queue** (lowest f first)

## Admissible Heuristics

### What is Admissibility?

**Admissible heuristic:** Never overestimates true cost to goal

```
h(n) ≤ true_cost(n, goal)  for all n
```

**Example - Manhattan Distance:**
```
n at (1, 2), goal at (4, 5)
Manhattan = |4-1| + |5-2| = 3 + 3 = 6
True cost ≥ 6 (can't go through walls)
Therefore: admissible! ✅
```

**Example - Inadmissible:**
```
h(n) = manhattan * 2
h = 12
True cost = 8
h > true cost ❌
NOT admissible - A* may not find optimal path!
```

### Why Admissibility Matters

**With admissible h:**
- ✅ A* finds optimal path (guaranteed)
- ✅ Explores efficiently
- ✅ Never misses better path

**With inadmissible h (h > true cost):**
- ❌ May find suboptimal path
- ✅ Explores even fewer nodes
- ❌ No optimality guarantee

**Trade-off:** Speed vs Optimality

## Heuristic Functions

### Common Heuristics for Grid Mazes

**1. Manhattan Distance (Default)**
```python
def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```
- Admissible for 4-way movement
- Exact for obstacle-free straight paths

**2. Euclidean Distance**
```python
def euclidean(pos1, pos2):
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
```
- Admissible for 8-way or free movement
- Too optimistic for 4-way (underestimates)

**3. Chebyshev Distance**
```python
def chebyshev(pos1, pos2):
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
```
- Admissible for 8-way movement
- Too optimistic for 4-way

**4. Zero Heuristic**
```python
def zero(pos1, pos2):
    return 0
```
- Always admissible
- A* becomes UCS (no guidance)

## Experimenting with Heuristics

### Using the Interactive Slider

1. Run: `python scripts/run_search.py`
2. Press `4` to start A*
3. Wait for completion, note "Nodes Expanded"
4. **Drag "Heuristic Weight" slider to 2.0**
5. Click **Apply**
6. A* restarts automatically!

**Compare:**
- Weight 1.0: Admissible, optimal path
- Weight 2.0: Inadmissible, fewer nodes, might be suboptimal

**Try weights:** 0.5, 1.0, 1.5, 2.0, 3.0

**Observe:**
- < 1.0: Too conservative, explores too much
- = 1.0: Perfect balance (optimal)
- > 1.0: Too aggressive, might miss optimal

### Creating Custom Heuristics

**Open `modules/search/heuristics.py`:**

```python
def my_custom_heuristic(pos1, pos2):
    # Your idea!
    manhattan = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Try these:
    # return manhattan * 1.5  # Inadmissible
    # return manhattan * 0.8  # Too conservative
    # return max(abs(...), abs(...))  # Chebyshev

    return manhattan
```

**Test it:**
```bash
python scripts/run_search.py
# Select A* with custom heuristic in code
```

## Comparing A* vs Others

### vs BFS
- **A***: Fewer nodes, same path
- **BFS**: More nodes, same path
- **Winner**: A* (more efficient)

### vs UCS
- **A***: Guided by heuristic, fewer nodes
- **UCS**: No heuristic, more nodes
- **Winner**: A* (when good heuristic available)

### vs Greedy
- **A***: Optimal path, slightly more nodes
- **Greedy**: Suboptimal path, fewer nodes
- **Winner**: Depends on needs (optimality vs speed)

### vs DFS
- **A***: Optimal, systematic
- **DFS**: Not optimal, may get lucky
- **Winner**: A* (almost always)

## Common Misconceptions

### ❌ "A* is always faster"
**Reality:** A* expands fewer nodes, but each expansion involves heuristic computation. Wall-clock time might be similar to BFS.

### ❌ "Any heuristic makes A* better"
**Reality:** Bad heuristics (inadmissible or unhelpful) can make A* worse than UCS.

### ❌ "Higher heuristic weight = always better"
**Reality:** Higher weight = faster but loses optimality. It's a trade-off.

### ❌ "A* never gets stuck"
**Reality:** A* is complete (finds solution if exists), but can take a long time in huge spaces.

## Advanced Topics

### Weighted A*

```python
f(n) = g(n) + w * h(n)  # w > 1
```

- w = 1: Standard A* (optimal)
- w > 1: Faster, suboptimal
- w = 0: UCS (no heuristic)

**Used when:** Speed more important than optimality

### Consistency (Monotonicity)

**Consistent heuristic:**
```
h(n) ≤ cost(n, n') + h(n')  for all n, n'
```

**Triangle inequality** - direct path never longer than indirect

**Why it matters:**
- Ensures A* never re-expands nodes
- More efficient
- Manhattan is consistent for grid worlds

### IDA* (Iterative Deepening A*)

Combines:
- A* guidance
- DFS memory efficiency
- Iterative deepening completeness

**Used when:** Memory extremely limited

## Exercises

### Exercise 1: Heuristic Competition

Create 3 different heuristics and test them:

```python
# In heuristics.py
def heuristic_1(pos1, pos2):
    return manhattan(pos1, pos2)

def heuristic_2(pos1, pos2):
    return manhattan(pos1, pos2) * 1.2

def heuristic_3(pos1, pos2):
    # Your creative idea!
    pass
```

**Measure:** Nodes expanded, path length, optimality

**Report:** Which is best? Why?

### Exercise 2: Admissibility Testing

**Task:** Design a heuristic and prove it's admissible

**Steps:**
1. Define your heuristic mathematically
2. Argue why h(n) ≤ true_cost for all n
3. Test empirically (run A*, check path is optimal)
4. Write up your proof

### Exercise 3: A* vs Greedy Analysis

**Run both on 10 different mazes:**

| Maze | A* Nodes | A* Path | Greedy Nodes | Greedy Path | Winner? |
|------|----------|---------|--------------|-------------|---------|
| 1    | ?        | ?       | ?            | ?           | ?       |
| ...  | ...      | ...     | ...          | ...         | ...     |

**Analyze:** When does Greedy fail? When does it work well?

## Real-World A* Applications

### Video Games (Most Common Use)
- **Starcraft**: Units navigate maps
- **Dragon Age**: Character pathfinding
- **The Sims**: Sim movement

**Why A*?** Real-time, optimal paths, handles dynamic obstacles

### GPS Navigation
- Google Maps, Waze
- Finds optimal routes
- Heuristic = straight-line distance

**Modified for:** Traffic, road types, time-dependent costs

### Robotics
- Warehouse robots (Amazon)
- Autonomous vehicles
- Drone delivery

**Challenges:** Dynamic environments, 3D space, uncertainty

## Summary

**A* is your go-to algorithm when:**
- ✅ You need optimal path
- ✅ You have a good heuristic
- ✅ You want efficiency
- ✅ Graph has weights

**Key to success:** **Good admissible heuristic!**

---

**Next:** [Designing Heuristics](Heuristic-Design) | [A* Code Walkthrough](A-Star-Code-Walkthrough)
