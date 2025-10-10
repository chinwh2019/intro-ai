# Getting Started

This guide will help you set up and run your first AI module.

## Prerequisites

- **Python 3.8 or higher** installed
- **Basic Python knowledge** (variables, functions, loops)
- **Terminal/command line** familiarity
- **Text editor** or IDE (VS Code, PyCharm, etc.)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/chinwh2019/intro-ai.git
cd intro-ai
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pygame` - For graphics and visualization
- `numpy` - For numerical operations (desktop only)
- `matplotlib` - For learning curves (desktop only)

### Step 3: Verify Installation

```bash
python scripts/run_search.py
```

If a window opens with a maze, you're all set! 🎉

## Running Your First Module

### Option 1: Search Algorithms (Easiest Start)

```bash
python scripts/run_search.py
```

**What you'll see:**
- A maze with green (start) and red (goal) cells
- Sidebar with statistics and controls
- Instructions at the top

**What to do:**
1. Press `1` to run BFS (Breadth-First Search)
2. Watch yellow cells spread outward (exploration)
3. When done, cyan cells show the path found
4. Press `R` to reset, try another algorithm!

### Option 2: Markov Decision Process

```bash
python scripts/run_mdp.py
```

**What you'll see:**
- A 5×5 grid world
- Gold cell (goal), red cell (danger)
- Gray cells (obstacles)

**What to do:**
1. Press `SPACE` to start value iteration
2. Watch cells change color (values updating)
3. Press `P` to see policy arrows
4. Arrows show optimal actions!

### Option 3: Reinforcement Learning

```bash
python scripts/run_rl.py --preset turbo
```

**What you'll see:**
- Snake game on the left
- Training statistics on the right
- The snake moving (initially randomly)

**What to do:**
1. Just watch! Training runs automatically
2. Notice epsilon (ε) decreasing
3. Watch average score improve
4. Press `SPACE` to pause anytime

**Be patient:** Snake looks dumb for first 100-200 episodes. This is normal learning!

## Alternative: Browser Version (No Installation!)

If you don't want to install anything, use the web version:

1. Visit: https://chinwh2019.github.io/intro-ai/
2. Click on a module card
3. Wait 10-20 seconds for it to load
4. Use same keyboard controls!

**Web version limitations:**
- No model save/load for RL
- No learning curve plots for RL
- Slightly slower performance
- But everything else works!

## Troubleshooting Installation

### "pip: command not found"

**Try:**
```bash
pip3 install -r requirements.txt
```

Or install pip:
```bash
# macOS/Linux
python3 -m ensurepip --upgrade

# Windows
python -m ensurepip --upgrade
```

### "pygame not found" Error

```bash
# Install pygame separately
pip install pygame

# Or on some systems
pip3 install pygame
```

### "No module named 'modules'"

**Make sure you're in the project root:**
```bash
cd intro-ai
ls  # Should see: modules/, scripts/, web/, README.md
```

### Python Version Too Old

**Check your version:**
```bash
python --version
```

Need Python 3.8+. If older, install from [python.org](https://www.python.org/downloads/)

### Permission Errors (macOS/Linux)

```bash
# Use pip with --user flag
pip install --user -r requirements.txt
```

### Windows: Missing Visual C++

If pygame installation fails on Windows, you may need:
- [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Or use pre-built wheels:
```bash
pip install pygame --only-binary :all:
```

## Verifying Everything Works

### Quick Test: All Modules

```bash
# Test search (should open window, close with Q)
python scripts/run_search.py
# Press Q to close

# Test MDP (should open window, close with ESC)
python scripts/run_mdp.py
# Press ESC to close

# Test RL (should open window, close with Q)
python scripts/run_rl.py --episodes 5
# Press Q to close after a few seconds
```

If all three work, you're ready! ✅

## Understanding the File Structure

```
intro-ai/
├── modules/              # Desktop implementations
│   ├── search/          # Search algorithms
│   ├── mdp/             # Markov Decision Processes
│   └── reinforcement_learning/  # Q-Learning
│
├── web/                 # Browser versions (Pygbag)
│   ├── search/
│   ├── mdp/
│   └── reinforcement_learning/
│
├── scripts/             # Runner scripts
│   ├── run_search.py
│   ├── run_mdp.py
│   └── run_rl.py
│
├── models/              # Saved models (RL module)
├── legacy/              # Old code (archived)
└── requirements.txt     # Dependencies
```

**Where to start coding:**
- Modify existing code: `modules/<module_name>/`
- Configuration: `modules/<module_name>/config.py`
- Run scripts: `scripts/run_*.py`

## Next Steps

### For Learning Algorithms
1. Read [Search Algorithms Overview](Search-Algorithms-Overview)
2. Try [BFS Tutorial](BFS-Tutorial)
3. Complete [Search Exercises](Search-Exercises)

### For Understanding Theory
1. Read [What is Search?](What-is-Search)
2. Read [What is an MDP?](What-is-MDP)
3. Read [What is Q-Learning?](What-is-Q-Learning)

### For Hands-on Practice
1. Follow [Code Walkthroughs](Code-Walkthroughs)
2. Try [Beginner Exercises](Beginner-Exercises)
3. Build [Custom Heuristics](Custom-Heuristics)

## Getting Help

1. **Check the wiki** - Search for your topic
2. **Read module READMEs** - Quick reference guides
3. **Try web version** - Sometimes easier than debugging installation
4. **Ask in class** - Instructor and TAs are here to help
5. **GitHub Issues** - Report bugs or request features

## Important Links

- **Repository**: https://github.com/chinwh2019/intro-ai
- **Web Demos**: https://chinwh2019.github.io/intro-ai/
- **Course Wiki**: https://github.com/chinwh2019/intro-ai/wiki

---

**Ready to start? Pick a module and dive in!** 🚀

**Recommended first module:** [Search Algorithms](Search-Algorithms-Overview) (easiest to understand)
