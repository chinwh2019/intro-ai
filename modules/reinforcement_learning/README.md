# Reinforcement Learning Module

Interactive Q-Learning implementation with Snake game environment featuring real-time learning visualization.

## Features

- **Q-Learning Algorithm**: Classic off-policy temporal difference learning
- **Snake Environment**: Challenging game with 11-dimensional state space
- **Real-time Visualization**: Watch Q-values, exploration, and learning curves
- **Interactive Controls**: Adjust learning rate, epsilon, discount, and speed on-the-fly
- **Training/Inference Modes**: Switch between learning and watching trained agent
- **Model Persistence**: Save and load trained Q-tables (desktop version only)
- **Preset Configurations**: Quick-start settings for different learning scenarios

## Quick Start

```bash
# Install dependencies
pip install pygame numpy matplotlib

# Run with default settings
python scripts/run_rl.py

# Or use a preset
python scripts/run_rl.py --preset fast_learning
python scripts/run_rl.py --preset turbo

# Load a trained model
python scripts/run_rl.py --load
```

## Controls

### Keyboard
- **SPACE**: Pause/Resume training
- **S**: Save current model
- **L**: Load saved model
- **T**: Toggle Training/Inference mode
- **Q**: Quit application

### UI Controls
- **Sliders**: Adjust learning rate, discount, epsilon, speed
- **Apply Button**: Apply slider changes
- **Preset Buttons**: Load predefined configurations (Default, Fast, Slow, Turbo)
- **Save/Load Buttons**: Manage trained models
- **Training Mode Toggle**: Switch between learning and watching

## Understanding the Visualization

### Game Area (Left Side)
- **Green block**: Snake head
- **Dark green**: Snake body
- **Red block**: Food
- **Grid**: Game environment (800×600 default)

### Stats Panel (Right Side)

**Episode Information:**
- Episode number / total episodes
- Current score (food eaten)
- Steps taken this episode

**Learning Parameters:**
- **ε (epsilon)**: Exploration rate (1.0 = 100% random, 0.01 = 1% random)
- **α (alpha)**: Learning rate (how fast Q-values update)
- **Q-table size**: Number of unique states encountered

**Q-Values (Current State):**
- Straight, Right, Left actions
- Green = best action
- Shows what agent has learned

**Recent Performance (Last 100 episodes):**
- Average score
- Max score achieved
- Average total reward

## Q-Learning Explained

### State Representation (11 features)

The agent observes:
1. **Danger detection** (3): Collision ahead? Collision right? Collision left?
2. **Current direction** (4): Moving up/right/down/left? (one-hot)
3. **Food location** (4): Food to left/right/up/down?

### Action Space (3 actions)
- **0**: Continue straight
- **1**: Turn right (90° clockwise)
- **2**: Turn left (90° counter-clockwise)

### Reward System
- **+10**: Ate food
- **-10**: Crashed (wall or self-collision)
- **+1**: Moved closer to food
- **-1.5**: Moved away from food
- **-0.1**: Idle movement

### Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- **Q(s,a)**: Value of taking action a in state s
- **α (alpha)**: Learning rate (0.01 default)
- **r**: Immediate reward
- **γ (gamma)**: Discount factor (0.95 default)
- **max Q(s',a')**: Best possible future value

### Exploration vs Exploitation

**ε-greedy policy:**
- With probability **ε**: Choose random action (explore)
- With probability **1-ε**: Choose best known action (exploit)
- **ε decays** over time: 1.0 → 0.01 (explore → exploit)

## Configuration

### Method 1: Use Presets (Easiest)

```python
from modules.reinforcement_learning.config import load_preset

load_preset('default')         # Balanced learning
load_preset('fast_learning')   # Faster convergence, speed 150
load_preset('slow_careful')    # Slower, more thorough, speed 20
load_preset('turbo')           # Maximum speed 200 for quick training
```

### Method 2: Modify Parameters

```python
from modules.reinforcement_learning.config import config

# Hyperparameters
config.LEARNING_RATE = 0.05        # Faster learning
config.DISCOUNT_FACTOR = 0.99      # Value future more
config.EPSILON_DECAY = 0.99        # Decay exploration faster

# Training
config.NUM_EPISODES = 500          # Train for 500 episodes
config.GAME_SPEED = 100            # Faster simulation

# Rewards (reward shaping)
config.REWARD_FOOD = 20.0          # Bigger reward for food
config.REWARD_CLOSER_TO_FOOD = 2.0 # Encourage approaching food
```

### Method 3: Command Line

```bash
python scripts/run_rl.py --episodes 500 --lr 0.05 --speed 150
```

## Learning Objectives

Students will understand:

- **Reinforcement Learning Paradigm**: Agent-environment interaction
- **Q-Learning Algorithm**: Off-policy TD learning
- **Exploration-Exploitation Tradeoff**: Balancing discovery vs using knowledge
- **Temporal Difference Learning**: Learning from prediction errors
- **Reward Shaping**: Designing rewards to guide learning
- **Hyperparameter Tuning**: Effect of learning rate, discount, epsilon
- **Convergence**: How Q-values stabilize over training

## Experiments for Students

### Beginner: Watch Learning Happen

1. Run: `python scripts/run_rl.py --preset slow_careful`
2. Watch the snake initially move randomly (high ε)
3. Observe Q-values appearing in the sidebar
4. Notice epsilon decreasing
5. See score improving over episodes

**Questions:**
- Why does the snake play randomly at first?
- When does it start playing better?
- What happens to epsilon over time?

### Intermediate: Hyperparameter Effects

**Experiment 1: Learning Rate**
```bash
python scripts/run_rl.py --lr 0.001 --episodes 200  # Slow
python scripts/run_rl.py --lr 0.1 --episodes 200    # Fast
```
**Question:** Which converges faster? Which is more stable?

**Experiment 2: Discount Factor**
```bash
python scripts/run_rl.py --discount 0.5   # Short-sighted
python scripts/run_rl.py --discount 0.99  # Far-sighted
```
**Question:** How does this affect the agent's strategy?

**Experiment 3: Reward Shaping**
```python
# Edit config.py
config.REWARD_CLOSER_TO_FOOD = 5.0   # Strong guidance
config.REWARD_FARTHER_FROM_FOOD = -5.0

# vs

config.REWARD_CLOSER_TO_FOOD = 0.0   # No guidance
config.REWARD_FARTHER_FROM_FOOD = 0.0
```
**Question:** Does reward shaping help or hurt learning?

### Advanced: Training Analysis

**Save a trained model:**
```bash
python scripts/run_rl.py --episodes 1000 --preset fast_learning
# During training, press 'S' to save
```

**Load and watch it play:**
```bash
python scripts/run_rl.py --load
# Press 'T' to enter inference mode
```

**Analyze Q-table:**
```python
import json
with open('models/q_table.json') as f:
    data = json.load(f)
print(f"States learned: {len(data['q_table'])}")
print(f"Episodes trained: {data['episodes_trained']}")
```

## File Structure

```
modules/reinforcement_learning/
├── README.md (this file)
├── config.py                 # Configuration and presets
├── core/
│   └── q_learning.py         # Q-Learning agent
├── environments/
│   ├── base_env.py           # Base environment interface
│   └── snake.py              # Snake game environment
├── ui/
│   ├── visualizer.py         # Main visualization
│   └── controls.py           # Interactive parameter panel
├── utils/
│   └── stats.py              # Training statistics
└── main.py                   # Main training loop
```

## Technical Details

**State Space:**
- 11 binary/continuous features
- Infinite possible states (continuous space)
- Q-table uses sparse dictionary representation

**Q-Table:**
- Dictionary: `state_tuple → [Q0, Q1, Q2]`
- Only stores visited states (memory efficient)
- Grows during training as new states are encountered

**Training Process:**
1. Start with empty Q-table (all zeros)
2. Agent explores randomly (high ε)
3. Q-values get updated after each step
4. Agent gradually exploits learned knowledge (decreasing ε)
5. Performance improves as Q-table becomes more accurate

**Typical Learning Curve:**
- Episodes 0-200: Random exploration, low scores (0-2)
- Episodes 200-500: Learning accelerates, scores improve (2-5)
- Episodes 500-1000: Refinement, stable performance (5-10+)

## Performance Expectations

**Training Speed:**
- Default (50 steps/sec): ~20 minutes for 1000 episodes
- Fast preset (150 steps/sec): ~7 minutes for 1000 episodes
- Turbo preset (200 steps/sec): ~5 minutes for 1000 episodes

**Expected Results:**
- Untrained agent: Average score 0-1
- After 500 episodes: Average score 3-5
- After 1000 episodes: Average score 5-10
- Well-tuned agent: Can reach 15-20+ occasionally

## For Instructors

### Classroom Activities

**Activity 1: Observe ε-Greedy** (15 min)
- Students watch initial random behavior
- Pause at episode 100, 500, 1000
- Discuss how epsilon decay affects learning

**Activity 2: Hyperparameter Hunt** (30 min)
- Groups try different learning rates
- Compare convergence speed
- Present findings to class

**Activity 3: Reward Engineering** (30 min)
- Modify reward structure
- Test if shaped rewards help or hurt
- Discuss reward design principles

### Assessment Ideas

**Conceptual Questions:**
- Why does Q-learning need both exploration and exploitation?
- What happens if α is too high? Too low?
- Why discount future rewards?

**Practical Assignment:**
- Train agent for 1000 episodes
- Document hyperparameters used
- Report final average score
- Analyze what worked and why

**Advanced Project:**
- Implement SARSA and compare with Q-learning
- Create new environment (different game)
- Implement experience replay

## Troubleshooting

**Issue: Agent not improving**
- Check epsilon is decaying (should decrease over episodes)
- Try higher learning rate (0.05 instead of 0.01)
- Verify rewards are working (watch "Total Reward" stat)

**Issue: Training too slow**
- Use `--preset turbo` for speed 200
- Reduce FPS: `config.FPS = 30`
- Run without visualization (advanced)

**Issue: Agent learns then gets worse**
- Learning rate might be too high (try 0.001)
- Epsilon might be decaying too fast (try 0.999)

## Web Version

A browser-compatible version is available at `web/reinforcement_learning/`:
- No installation required
- Runs in any modern browser
- Limitations: No model save/load, no matplotlib learning curves
- Access via GitHub Pages deployment

## License

MIT License - See root LICENSE file
