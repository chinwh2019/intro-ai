# Search Module - Student Quick Guide

## 🚀 How to Run

```bash
# Basic - just run it!
python scripts/run_search.py

# With a preset
python scripts/run_search.py --preset simple
python scripts/run_search.py --preset fast
```

Once running:
- Press **1-5** to select algorithm (1=BFS, 2=DFS, 3=UCS, 4=A*, 5=Greedy)
- Press **SPACE** to pause/resume
- Press **S** to step through one iteration (when paused)
- Press **T** to toggle random start/goal mode (see indicator in sidebar!)
- Press **R** to reset and generate a new maze
- Press **Q** to quit

**NEW:** Use the **interactive sliders** at the bottom left to adjust:
- **Speed**: Make search faster/slower in real-time
- **Heuristic Weight**: Experiment with A* and Greedy heuristics
- **Complexity**: Change how many walls appear in the next maze

**TIP:** Press **T** then **R** multiple times to get different start/goal positions!

---

## 🎮 Your First Experiment

**Activity: Compare BFS and A***

1. Run: `python scripts/run_search.py`
2. Press `1` to run BFS - watch it explore
3. Look at "Nodes Expanded" in the sidebar
4. Press `R` to reset (keeps same maze)
5. Press `4` to run A* - watch it explore
6. Compare: Which expanded fewer nodes? Why?

**Bonus:** Use the **speed slider** at the bottom left:
- Drag slider to change speed
- Click **Apply** button
- Watch the current search speed up or slow down!

---

## ⚙️ Changing Settings (No coding needed!)

### Method 1: Use the Interactive Sliders (Easiest!)

**While the program is running:**

1. Look at the **bottom left corner** - you'll see 3 sliders
2. **Speed Slider**: Drag to change animation speed (0.1x to 10x)
   - Left = slower (0.1x - good for understanding)
   - Right = faster (10x - good for testing)
3. **Heuristic Weight Slider**: Drag to experiment with A*/Greedy (0.5 to 3.0)
   - 1.0 = optimal (admissible)
   - >1.0 = faster but may not find shortest path
4. **Complexity Slider**: Drag to change wall density (0.3 to 1.0)
   - 0.3 = few walls (easier mazes)
   - 1.0 = many walls (harder mazes)
5. Click **Apply** button to activate changes!

**Tips:**
- Speed changes work immediately on current search
- Heuristic weight automatically restarts A*/Greedy with new setting
- Complexity takes effect when you press **R** to reset

### Method 2: Use Presets (Command Line)

```bash
python scripts/run_search.py --preset simple      # Small 20x15 maze
python scripts/run_search.py --preset large_maze  # Big 60x40 maze
```

### Method 3: Custom Command Line

```bash
python scripts/run_search.py --speed 5.0   # 5x faster
python scripts/run_search.py --speed 0.5   # 2x slower
python scripts/run_search.py --width 50 --height 35  # Custom size
```

---

## 🎨 Making the Start/Goal Different Each Time

**Method 1: Use the T Key (Easiest!)**

1. Run the program: `python run_search.py`
2. Press **T** to enable random mode (watch sidebar turn green!)
3. Press **R** to reset - you'll see different start/goal positions!
4. Press **R** again - they'll be different again!

The sidebar shows: **Random: ON** (in green) or **Random: OFF**

**Method 2: Set in Code**

Edit `scripts/run_search.py` and add this before `main()`:

```python
from modules.search.config import config
config.RANDOM_START_GOAL = True  # Start with random mode ON
```

**Method 3: Set Specific Positions**

```python
# In scripts/run_search.py, add before main()
from modules.search.config import config
config.START_POSITION = (10, 10)  # Row 10, Column 10
config.GOAL_POSITION = (25, 35)   # Row 25, Column 35
```

Now start and goal will always be at these exact positions!

---

## 🎛️ Using the Interactive Controls (Super Easy!)

### Experiment: Make A* Inadmissible

**What happens when heuristic weight > 1.0?**

1. Run: `python scripts/run_search.py`
2. Press `4` to start A*
3. Wait for it to finish, note "Nodes Expanded"
4. Look at **bottom left** - find the "Heuristic Weight" slider
5. **Drag it to 2.0** (double the heuristic)
6. Click **Apply**
7. A* automatically restarts with new weight!

**Compare:**
- Heuristic weight 1.0: Optimal path, more nodes
- Heuristic weight 2.0: Faster search, but might not be optimal!

**Question:** Did path length change? Did nodes expanded change?

---

### Experiment: Speed Control

**Watch search in slow motion:**

1. Start any algorithm (press 1-5)
2. While it's running, drag **Speed slider** to 0.3
3. Click **Apply**
4. Watch it slow down immediately!

**Then speed it up:**
5. Drag Speed slider to 5.0
6. Click Apply
7. Watch it zoom!

**Use this to:** Understand what algorithm is doing (slow), or test quickly (fast)

---

### Experiment: Maze Complexity

**Make mazes easier or harder:**

1. Drag **Complexity slider** to 0.3 (easy - few walls)
2. Click **Apply**
3. Press **R** to reset - new maze will have fewer walls!
4. Now drag **Complexity slider** to 0.9 (hard - many walls)
5. Click **Apply**
6. Press **R** again - maze is much harder!

**Question:** Do algorithms perform differently on easy vs hard mazes?

---

## 🧠 Creating Your Own Heuristic (Intermediate)

**Step 1:** Open `modules/search/heuristics.py`

**Step 2:** Find the `my_custom_heuristic` function (around line 94)

**Step 3:** Modify it! Try different ideas:

```python
def my_custom_heuristic(pos1, pos2):
    # Your idea here!
    manhattan = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Try these experiments:
    # return manhattan * 2.0          # Double weight (faster but not optimal)
    # return manhattan * 0.5          # Half weight (still optimal, less informed)
    # return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))  # Chebyshev

    return manhattan  # Start with this
```

**Step 4:** Test it in terminal:

```python
python3 -c "
from modules.search.core.environment import Maze
from modules.search.algorithms.astar import AStar
from modules.search.heuristics import my_custom_heuristic

maze = Maze(30, 20)
astar = AStar(maze, heuristic='custom')  # Uses your custom heuristic!

for _ in astar.search():
    pass

print(f'Nodes expanded: {astar.nodes_expanded}')
print(f'Path length: {len(astar.solution_path)}')
"
```

**Step 5:** Compare with Manhattan to see if yours is better!

---

## 📊 Running the Examples

We created 8 ready-to-run examples for you:

```bash
python modules/search/examples.py
```

Then choose:
1. **Use presets** - See how presets work
2. **Custom configuration** - Modify parameters
3. **Custom start/goal** - Set specific positions
4. **Custom heuristics** - Use different heuristics
5. **Create your heuristic** - Template for making your own
6. **Compare all algorithms** - Side-by-side comparison
7. **Test admissibility** - Check if heuristic is admissible
8. **Full features demo** - Everything together

---

## 🎯 Challenges

### Challenge 1: Find the Fastest Configuration
Find the settings that make A* solve mazes the fastest.
- Try different heuristics
- Try different weights
- Measure nodes expanded

### Challenge 2: Design a Better Heuristic
Can you create a heuristic that:
- Expands fewer nodes than Manhattan distance?
- Still finds the optimal path?
(Hint: It must be admissible!)

### Challenge 3: When Does DFS Beat BFS?
Find a maze where DFS expands fewer nodes than BFS.
- Experiment with different maze sizes
- Try different complexities
- Explain why this happens

---

## 🔬 Advanced: Coding Your Own Algorithm

**Open `modules/search/algorithms/` and create `my_algorithm.py`:**

```python
from modules.search.core.base_algorithm import SearchAlgorithm
from modules.search.core.state import State, Node

class MyAlgorithm(SearchAlgorithm):
    def search(self):
        # Your algorithm here!
        # See bfs.py or dfs.py for examples
        pass
```

Then add it to `main.py` to run it in the visualization!

---

## 💡 Tips

- **Start simple**: Use `--preset simple` for faster testing
- **Step through**: Use `S` key to understand algorithm behavior
- **Compare**: Run multiple algorithms on the same maze
- **Experiment**: Change one parameter at a time
- **Measure**: Always check nodes expanded and path length

---

## ❓ Common Questions

**Q: Why doesn't my preset work?**
A: Make sure to restart the program after loading a preset, or use command line: `python scripts/run_search.py --preset simple`

**Q: How do I make my own heuristic?**
A: Edit `modules/search/heuristics.py` and modify `my_custom_heuristic`, or create a new function!

**Q: Can I make diagonal movements?**
A: Not in the basic version, but this is a great advanced project! You'd need to modify `get_neighbors()` in environment.py

**Q: Why is A* sometimes slower than BFS?**
A: A* is optimal AND efficient, but "efficient" means expanding fewer nodes, not necessarily faster runtime (heuristic calculation takes time)

---

## 📚 Learning Goals

After completing these activities, you should understand:
- ✅ How search algorithms explore state spaces
- ✅ Difference between uninformed (BFS, DFS) and informed (A*, Greedy) search
- ✅ Trade-offs: completeness, optimality, time, space
- ✅ How heuristics guide search
- ✅ What makes a heuristic admissible
- ✅ When to use which algorithm

---

## 🌐 Want to Try It Without Installing?

Your instructor may have deployed this to the web! Ask for the link.

**Web version features:**
- ✅ All same features (5 algorithms, interactive sliders, random mode)
- ✅ Works on Chromebooks, tablets, any browser
- ✅ No Python installation needed
- ✅ Perfect for trying at home

**If you have the link, just:**
1. Open it in your browser
2. Wait 5-10 seconds for it to load
3. Press 1-5 to select algorithm
4. Everything works the same!

---

**Have fun exploring! 🎉**
