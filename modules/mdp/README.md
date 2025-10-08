# Markov Decision Process (MDP) Module

Interactive visualization of planning under uncertainty using value iteration and policy iteration.

## Features

- **Value Iteration Algorithm**: Watch Bellman backups propagate through the state space
- **Interactive Grid World**: Obstacles, goal, danger, and stochastic transitions
- **Real-time Visualization**: See values converge and optimal policy emerge
- **Configurable Parameters**: Experiment with discount, noise, and rewards
- **Multiple Views**: Toggle values, policy arrows, and Q-values

## Quick Start

```bash
# Install dependencies
pip install pygame

# Run the module
python run_mdp.py

# Or use a preset
python run_mdp.py --preset deterministic
python run_mdp.py --preset high_noise
```

## Controls

- **SPACE**: Start/Pause value iteration
- **S**: Step through one iteration (when paused)
- **R**: Reset grid world (new random layout)
- **V**: Toggle value display
- **P**: Toggle policy arrows
- **Q**: Toggle Q-values display

## Understanding the Visualization

### Colors

- **Gold**: Goal state (reward: +1.0)
- **Red**: Danger state (reward: -1.0)
- **Gray**: Obstacles (cannot enter)
- **Green/Red gradient**: Value function (green = positive, red = negative)
- **Blue circle**: Starting position
- **White arrows**: Optimal policy (best action in each state)

### Statistics Panel

- **Iteration**: Current iteration number
- **Discount (γ)**: How much future rewards are valued (0.9 = 90%)
- **Noise**: Probability of unintended transitions (0.2 = 20% chance)
- **Living Reward**: Cost per step (-0.04)

### What You're Watching

**Value Iteration:**
1. Starts with all values = 0
2. Each iteration updates values using Bellman equation
3. Values propagate from goal/danger outward
4. Converges when changes < 0.001
5. Optimal policy emerges automatically

## Configuration

### Method 1: Use Presets

```python
from modules.mdp.config import load_preset

load_preset('deterministic')   # No noise (always moves as intended)
load_preset('high_noise')      # 40% noise (more unpredictable)
load_preset('cliff_world')     # Large grid, high living cost
load_preset('fast_convergence')  # Fast animation for quick demos
```

### Method 2: Modify Parameters

```python
from modules.mdp.config import config

# Make environment more stochastic
config.NOISE = 0.4

# Value future rewards more
config.DISCOUNT = 0.95

# Make living more expensive
config.LIVING_REWARD = -0.1

# Larger grid
config.GRID_SIZE = 8
```

### Method 3: Command Line

```bash
python run_mdp.py --grid-size 8 --noise 0.3 --discount 0.95
```

## Learning Objectives

Students will understand:

- **MDP Components**: States, actions, transitions, rewards, discount
- **Bellman Equation**: How optimal values are computed
- **Value Function**: Expected total reward from each state
- **Policy**: Optimal action in each state
- **Convergence**: How values stabilize over iterations
- **Stochastic Transitions**: Dealing with uncertainty
- **Discount Factor**: Balancing immediate vs future rewards

## Experiments for Students

### Beginner: Observe Convergence

1. Press SPACE to start
2. Watch values spread from goal/danger
3. Notice how green (positive) values spread from goal
4. Notice how red (negative) values spread from danger
5. Press P to see optimal policy arrows

**Question:** Why do values near the goal become green first?

### Intermediate: Noise Effects

```python
# Try with no noise
config.NOISE = 0.0
# Run - policy goes straight to goal

# Try with high noise
config.NOISE = 0.5
# Run - policy might be more conservative
```

**Question:** How does noise affect the optimal policy?

### Advanced: Discount Factor

```python
for discount in [0.5, 0.9, 0.99]:
    config.DISCOUNT = discount
    # Run and observe policy changes
```

**Question:** Why does low discount make agent "short-sighted"?

## File Structure

```
modules/mdp/
├── README.md (this file)
├── config.py                 # Configuration with presets
├── core/
│   ├── mdp.py               # MDP class definition
│   └── solver.py            # Value iteration, policy iteration
├── environments/
│   └── grid_world.py        # Grid world with obstacles
├── ui/
│   └── visualizer.py        # Pygame visualization
└── main.py                  # Main application
```

## Technical Details

**MDP Formulation:**
- **States**: Grid cells (5x5 = 25 states)
- **Actions**: {UP, DOWN, LEFT, RIGHT}
- **Transitions**: Stochastic (80% intended, 10% each perpendicular)
- **Rewards**: +1 (goal), -1 (danger), -0.04 (living cost)
- **Discount**: 0.9 (future rewards worth 90% of current)

**Value Iteration:**
- Iteratively applies Bellman optimality equation
- Q(s,a) = R(s,a) + γ Σ T(s,a,s') V(s')
- V(s) = max_a Q(s,a)
- Converges to optimal value function

**Policy:**
- Derived from values: π(s) = argmax_a Q(s,a)
- Shows best action in each state
- Visualized as arrows pointing in optimal direction

## For Instructors

### Classroom Activities

**Activity 1: Value Propagation** (15 min)
- Students watch value iteration
- Pause and discuss why certain cells update first
- Connect to dynamic programming concepts

**Activity 2: Parameter Exploration** (20 min)
- Students modify discount factor
- Observe policy changes
- Understand time preference

**Activity 3: Stochastic vs Deterministic** (20 min)
- Compare noise=0.0 vs noise=0.4
- Discuss real-world uncertainty
- See how policy adapts

### Assessment Ideas

**Quiz Questions:**
- What happens to the policy if discount = 0?
- Why might an optimal policy avoid being near danger?
- How does noise affect optimal path?

**Coding Assignment:**
- Modify grid world to add more obstacles
- Implement custom reward structure
- Implement policy iteration and compare

## License

MIT License - See root LICENSE file
