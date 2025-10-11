# Introduction to Artificial Intelligence

Interactive visualizations and implementations of fundamental AI algorithms for education and experimentation. All modules available as **desktop applications** and **browser-based demos** (no installation required!).

## 🌟 Features

- **Interactive Visualizations** - Watch algorithms work step-by-step
- **Browser & Desktop** - Run locally or access via web
- **Student-Friendly** - Comprehensive guides and tutorials
- **Configurable** - Adjust parameters with sliders and presets
- **Educational Focus** - Built for learning, not just demonstration

## 🚀 Quick Start

### Try in Your Browser (No Installation!)

Visit the live demos:
- **Search Algorithms**: https://chinwh2019.github.io/intro-ai/search/
- **MDP**: https://chinwh2019.github.io/intro-ai/mdp/
- **Reinforcement Learning**: https://chinwh2019.github.io/intro-ai/reinforcement_learning/

### Run on Your Computer

```bash
# Clone the repository
git clone https://github.com/chinwh2019/intro-ai.git
cd intro-ai

# Install dependencies
pip install -r requirements.txt

# Run any module
python scripts/run_search.py
python scripts/run_mdp.py
python scripts/run_rl.py
```

## 📚 Documentation

- **[GitHub Wiki](https://github.com/chinwh2019/intro-ai/wiki)** - Comprehensive tutorials and theory
- **Module READMEs** - Quick reference for each module
- **Student Guides** - Step-by-step learning paths

## 📂 Project Structure

```
intro-ai/
├── modules/              # Desktop implementations
│   ├── search/          # Search algorithms (BFS, DFS, UCS, A*, Greedy)
│   ├── mdp/             # Markov Decision Processes
│   └── reinforcement_learning/  # Q-Learning with Snake
│
├── web/                 # Browser-compatible versions (Pygbag/WASM)
│   ├── search/
│   ├── mdp/
│   └── reinforcement_learning/
│
├── scripts/             # Runner scripts
│   ├── run_search.py
│   ├── run_mdp.py
│   └── run_rl.py
│
├── wiki/                # GitHub Wiki content
├── legacy/              # Archived old implementations
└── requirements.txt     # Dependencies
```

## 🔍 Search Algorithms

Interactive maze solver with 5 search algorithms:

- **Breadth-First Search (BFS)** - Layer-by-layer exploration
- **Depth-First Search (DFS)** - Deep exploration with backtracking
- **Uniform Cost Search (UCS)** - Optimal for weighted graphs
- **A* Search** - Optimal + efficient with heuristics
- **Greedy Best-First** - Fast heuristic-guided search

**Features:**
- Real-time step-by-step visualization
- Interactive parameter controls (speed, heuristic weight, complexity)
- Performance metrics comparison
- Random maze generation
- Custom start/goal positioning

**Documentation:** [modules/search/README.md](modules/search/README.md) | [Student Guide](modules/search/STUDENT_GUIDE.md)

## 🎲 Markov Decision Processes (MDPs)

Grid world environment with planning under uncertainty:

- **Value Iteration** - Compute optimal values and policies
- **Stochastic Transitions** - Model uncertainty (slippery grid)
- **Interactive Controls** - Adjust discount, noise, rewards
- **Visual Convergence** - Watch values propagate

**Features:**
- Real-time value function visualization
- Policy arrows showing optimal actions
- Q-value display for all state-action pairs
- Configurable discount factor and noise levels

**Documentation:** [modules/mdp/README.md](modules/mdp/README.md) | [Student Guide](modules/mdp/STUDENT_GUIDE.md)

## 🐍 Reinforcement Learning

Q-Learning agent learning to play Snake:

- **Q-Learning Algorithm** - Off-policy TD learning
- **ε-Greedy Exploration** - Balanced exploration/exploitation
- **Interactive Training** - Adjust hyperparameters in real-time
- **Training/Inference Modes** - Learn or watch trained agent
- **Model Persistence** - Save and load trained Q-tables (desktop)

**Features:**
- Live Q-value display showing agent's learning
- Training statistics and learning curves
- Preset configurations for different learning scenarios
- Parameter sliders (learning rate, epsilon, discount, speed)

**Documentation:** [modules/reinforcement_learning/README.md](modules/reinforcement_learning/README.md) | [Student Guide](modules/reinforcement_learning/STUDENT_GUIDE.md)

## 🤖 Machine Learning

[Coming Soon]

## ✨ Generative AI

[Coming Soon]

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/chinwh2019/intro-ai.git
cd intro-ai

# 2. Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- `pygame` - Graphics and visualization
- `numpy` - Numerical operations (desktop only)
- `matplotlib` - Learning curves (desktop only, RL module)

## 🎮 Usage

### Desktop Versions (Full Features)

```bash
# Search Algorithms
python scripts/run_search.py
# Press 1-5 to select algorithm, SPACE to pause, R to reset

# MDP
python scripts/run_mdp.py
# Press SPACE to start value iteration, P for policy, V for values

# Reinforcement Learning
python scripts/run_rl.py --preset turbo
# Press T to toggle training/inference, S to save model
```

### With Presets

```bash
# Fast learning
python scripts/run_rl.py --preset fast_learning

# Simple small maze
python scripts/run_search.py --preset simple

# Deterministic MDP (no noise)
python scripts/run_mdp.py --preset deterministic
```

### Browser Versions (No Installation)

Visit the live demos:
- Search: https://chinwh2019.github.io/intro-ai/search/
- MDP: https://chinwh2019.github.io/intro-ai/mdp/
- RL: https://chinwh2019.github.io/intro-ai/reinforcement_learning/

**Note:** Web versions don't support model save/load or matplotlib visualizations.

## 📖 Learning Resources

### For Students

1. **[GitHub Wiki](https://github.com/chinwh2019/intro-ai/wiki)** - Comprehensive theory and tutorials
   - Algorithm explanations
   - Step-by-step guides
   - Exercises and challenges
   - Troubleshooting

2. **Module Documentation**
   - [Search README](modules/search/README.md) + [Student Guide](modules/search/STUDENT_GUIDE.md)
   - [MDP README](modules/mdp/README.md) + [Student Guide](modules/mdp/STUDENT_GUIDE.md)
   - [RL README](modules/reinforcement_learning/README.md) + [Student Guide](modules/reinforcement_learning/STUDENT_GUIDE.md)

3. **Interactive Controls**
   - All modules have parameter sliders
   - Adjust settings in real-time
   - Experiment without coding

### For Instructors

- Modular architecture for easy customization
- Preset configurations for classroom demos
- Web deployment for zero-setup student access
- Comprehensive wiki for reducing repetitive questions

## 🎯 Key Features by Module

### Search Algorithms
- ✅ 5 algorithms (BFS, DFS, UCS, A*, Greedy)
- ✅ Interactive parameter panel (speed, heuristic, complexity)
- ✅ Step-by-step execution
- ✅ Random start/goal mode
- ✅ Performance metrics

### MDP
- ✅ Value iteration visualization
- ✅ Policy display with arrows
- ✅ Q-value display for all actions
- ✅ Stochastic transitions (configurable noise)
- ✅ Real-time convergence tracking

### Reinforcement Learning
- ✅ Q-Learning with ε-greedy
- ✅ Live Q-value display
- ✅ Training/inference modes
- ✅ Model save/load (desktop)
- ✅ Interactive hyperparameter tuning
- ✅ Learning curve visualization (desktop)

## 🛠️ Configuration

All modules support multiple configuration methods:

**Method 1: Interactive sliders** (easiest - while running)
- Adjust parameters via UI
- Click Apply button
- Changes take effect immediately

**Method 2: Presets** (command line)
```bash
python scripts/run_rl.py --preset turbo
python scripts/run_search.py --preset simple
```

**Method 3: Edit config.py** (programmatic)
```python
from modules.search.config import config
config.MAZE_WIDTH = 50
config.ANIMATION_SPEED = 2.0
```

## 🌐 Web Deployment

All modules deployed to GitHub Pages using Pygbag (Python → WebAssembly):

- **Automatic deployment** via GitHub Actions
- **No server costs** - static file hosting
- **Instant updates** - push to main, auto-deploys
- **Cross-platform** - works on any modern browser

**Architecture:**
- Desktop: Full Python with NumPy, Matplotlib
- Web: Pure Python (NumPy-free), async/await compatible

## 🤝 Contributing

Contributions welcome! Areas where help is appreciated:

- New algorithms (bidirectional search, IDA*, SARSA, etc.)
- New environments (different games, puzzles)
- Additional exercises and tutorials
- Wiki page improvements
- Bug fixes and optimizations

**Process:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Commit: `git commit -m 'Add feature'`
5. Push: `git push origin feature-name`
6. Create Pull Request

## 🎓 For Educators

This repository is designed for teaching:

- **Modular design** - Use one or all modules
- **Multiple access modes** - Desktop, web, or both
- **Preset configurations** - Quick demos without coding
- **Comprehensive docs** - Wiki, READMEs, student guides
- **Active learning** - Students experiment, not just watch

**Classroom tested** - Used in university AI courses.

## 🔗 Links

- **Live Demos**: https://chinwh2019.github.io/intro-ai/
- **Wiki**: https://github.com/chinwh2019/intro-ai/wiki
- **Issues**: https://github.com/chinwh2019/intro-ai/issues

## 📝 License
