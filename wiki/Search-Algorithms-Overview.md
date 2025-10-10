# Search Algorithms - Overview

Search algorithms find paths through state spaces - from where you are to where you want to be.

## What is Search?

**Real-world example:**
- You're at point A, want to reach point B
- Many possible paths exist
- Some paths are shorter, some longer
- Search algorithms find a path (ideally the best one)

**In our maze:**
- **State space**: All possible positions in the maze
- **Start state**: Green cell (where you begin)
- **Goal state**: Red cell (where you want to reach)
- **Actions**: Move up, down, left, right
- **Path**: Sequence of moves from start to goal

## The Five Algorithms

### Uninformed Search (Blind Search)
Don't use knowledge about goal location:

1. **[Breadth-First Search (BFS)](BFS-Tutorial)**
   - Explores level by level
   - Finds shortest path
   - Uses lots of memory
   - **Best for:** Unweighted graphs, guaranteed shortest path

2. **[Depth-First Search (DFS)](DFS-Tutorial)**
   - Explores deeply before backtracking
   - Path may not be shortest
   - Uses less memory
   - **Best for:** Finding any solution quickly, deep search trees

3. **[Uniform Cost Search (UCS)](UCS-Tutorial)**
   - Expands least-cost path first
   - Finds optimal path with weighted edges
   - **Best for:** Weighted graphs, different move costs

### Informed Search (Uses Heuristics)
Uses knowledge about goal location to guide search:

4. **[A* Search](A-Star-Explained)**
   - Best of both worlds: optimal AND efficient
   - Uses heuristic + actual cost
   - **Best for:** Most pathfinding problems (games, robotics, GPS)

5. **[Greedy Best-First Search](Greedy-Tutorial)**
   - Only uses heuristic (ignores path cost)
   - Fast but not optimal
   - **Best for:** When speed matters more than optimality

## Visual Comparison

### How They Explore

**BFS:** Expands in concentric circles
```
        G
    Y Y R Y Y
  Y Y Y R Y Y Y
Y Y Y Y S Y Y Y Y
  Y Y Y R Y Y Y
    Y Y R Y Y
```

**DFS:** Goes deep in one direction
```
        G
        R
        R
Y Y Y Y S
```

**A*:** Aims toward goal
```
        G
      R R
    Y R R
  Y Y Y S
```

(Y = Explored, R = Frontier, S = Start, G = Goal)

## Performance Comparison

| Algorithm | Complete? | Optimal? | Time | Space | Best Use Case |
|-----------|-----------|----------|------|-------|---------------|
| **BFS** | ✅ Yes | ✅ Yes* | O(b^d) | O(b^d) | Unweighted graphs |
| **DFS** | ✅ Yes** | ❌ No | O(b^m) | O(bm) | Memory limited |
| **UCS** | ✅ Yes | ✅ Yes | O(b^d) | O(b^d) | Weighted graphs |
| **A*** | ✅ Yes | ✅ Yes*** | O(b^d) | O(b^d) | Most problems |
| **Greedy** | ❌ No | ❌ No | O(b^m) | O(b^m) | Speed critical |

*For unweighted graphs only
**In finite spaces
***With admissible heuristic

**Legend:**
- b = branching factor (avg children per node)
- d = depth of shallowest goal
- m = maximum depth

## Key Concepts

### 1. State Space
- **State**: A configuration of the world (e.g., a position in the maze)
- **State space**: All possible states
- **State transition**: Moving from one state to another

### 2. Search Tree
- **Node**: Represents a state and path to reach it
- **Parent**: The node we came from
- **Children**: States reachable in one action
- **Path**: Sequence of nodes from start to current

### 3. Frontier vs Explored
- **Frontier**: States we know about but haven't explored yet
- **Explored**: States we've already examined
- **Algorithm determines**: Which frontier node to expand next

### 4. Completeness
- **Complete**: Guaranteed to find solution if one exists
- BFS, UCS, A* are complete
- Greedy is not complete (can get stuck in loops)

### 5. Optimality
- **Optimal**: Finds the best (shortest/cheapest) solution
- BFS (unweighted), UCS, A* (admissible h) are optimal
- DFS and Greedy are not optimal

### 6. Heuristic Function
- **h(n)**: Estimated cost from n to goal
- **Admissible**: Never overestimates true cost
- **Consistent**: h(n) ≤ cost(n,n') + h(n')
- **Good heuristic**: Close to true cost, easy to compute

## Watching Algorithms in Action

### What to Observe

**When you run BFS:**
1. Yellow cells spread evenly in all directions
2. Forms concentric "rings" around start
3. Stops when red (goal) is reached
4. Path traced back from goal to start (cyan)

**When you run A*:**
1. Yellow cells bias toward goal direction
2. Forms elongated shape pointing at goal
3. Explores fewer cells than BFS
4. Finds same shortest path (if h is admissible)

**When you run Greedy:**
1. Yellow cells strongly bias toward goal
2. May ignore obstacles initially
3. Might explore fewer cells than A*
4. Path may be suboptimal

### Performance Metrics to Compare

After running each algorithm, check sidebar:
- **Nodes Expanded**: Fewer is better (efficiency)
- **Path Length**: Shorter is better (optimality)
- **Max Frontier**: Smaller is better (memory usage)

**Expected results on default 40×30 maze:**
- BFS: ~400 nodes, path length 50
- DFS: ~200 nodes, path length 150 (not optimal!)
- A*: ~150 nodes, path length 50 (optimal!)
- Greedy: ~80 nodes, path length 55 (fast but suboptimal)

## Common Visualizations Explained

### Colors Mean:
- 🟢 **Green cell**: Start position
- 🔴 **Red cell**: Goal position
- 🟡 **Yellow cells**: Already explored (visited)
- 🟣 **Purple cells**: Frontier (candidates to explore next)
- 🔵 **Cyan cells**: Solution path (start → goal)
- 🟠 **Orange cell**: Currently being explored
- ⬜ **Gray cells**: Walls (obstacles)

### Statistics Mean:
- **Nodes Expanded**: How much work the algorithm did
- **Path Length**: How long the solution is
- **Steps**: Total iterations
- **Solution Found**: Whether goal was reached

## Interactive Features

### Parameter Panel (Bottom Left)

**Speed Slider (0.1x to 10x):**
- Controls animation speed
- 0.1x = very slow (good for understanding)
- 10x = very fast (good for testing)

**Heuristic Weight (0.5 to 3.0):**
- Only affects A* and Greedy
- 1.0 = admissible (optimal)
- >1.0 = inadmissible (faster, maybe suboptimal)
- <1.0 = too conservative (slower)

**Complexity (0.3 to 1.0):**
- Controls wall density
- 0.3 = easy mazes (few walls)
- 1.0 = hard mazes (many walls)
- Takes effect on next reset (R key)

**Apply Button:**
- Click after adjusting sliders
- Changes take effect immediately

## Choosing the Right Algorithm

### Decision Tree

**Do you know edge costs?**
- No (all moves cost the same) → Use **BFS** or **A***
- Yes (different costs) → Use **UCS** or **A***

**Do you have a good heuristic?**
- Yes → Use **A*** (best choice!)
- No → Use **BFS** or **UCS**

**Is memory very limited?**
- Yes → Use **DFS** (warning: not optimal)
- No → Use **BFS** or **A***

**Need solution fast, optimality not critical?**
- Yes → Use **Greedy** or **DFS**
- No → Use **A*** or **BFS**

**For most problems: Use A* with a good heuristic!**

## Learning Path

### Week 1: Fundamentals
1. Read [What is Search?](What-is-Search)
2. Try [BFS Tutorial](BFS-Tutorial)
3. Complete [Beginner Exercises](Beginner-Exercises)

### Week 2: Uninformed Search
1. Read [DFS Tutorial](DFS-Tutorial)
2. Read [UCS Tutorial](UCS-Tutorial)
3. Compare all three algorithms
4. Complete [Uninformed Search Exercises](Uninformed-Search-Exercises)

### Week 3: Informed Search
1. Read [Heuristic Functions](Heuristic-Functions)
2. Read [A* Explained](A-Star-Explained)
3. Try [Greedy Tutorial](Greedy-Tutorial)
4. Complete [Informed Search Exercises](Informed-Search-Exercises)

### Week 4: Advanced
1. Design [Custom Heuristics](Custom-Heuristics)
2. Try [Advanced Challenges](Advanced-Search-Challenges)
3. Implement [New Algorithms](Implementing-Algorithms)

## Real-World Applications

### Where Search Algorithms are Used

**A* in Practice:**
- 🎮 Video game pathfinding (NPCs navigating maps)
- 🚗 GPS navigation (finding optimal routes)
- 🤖 Robot motion planning (warehouse robots)
- 🧩 Puzzle solving (15-puzzle, Rubik's cube)

**BFS in Practice:**
- 🌐 Web crawling (exploring websites)
- 📱 Social networks (degrees of separation)
- 🗺️ Network routing (shortest hop count)

**DFS in Practice:**
- 🔍 File system search (finding files)
- 🧩 Maze generation (creating mazes)
- 🔗 Cycle detection (finding loops in graphs)

## Next Topics

After mastering search, explore:
- **[Markov Decision Processes](MDP-Overview)** - Planning with uncertainty
- **[Reinforcement Learning](Reinforcement-Learning-Overview)** - Learning through interaction

---

**Start exploring:** [BFS Tutorial](BFS-Tutorial) →
