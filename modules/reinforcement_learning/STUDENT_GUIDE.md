# Reinforcement Learning - Student Quick Guide

## 🚀 How to Run

```bash
# Basic - just run it!
python scripts/run_rl.py

# Fast training (150 steps/second)
python scripts/run_rl.py --preset fast_learning

# Super fast training (200 steps/second, 5 minutes for 1000 episodes!)
python scripts/run_rl.py --preset turbo

# Train for specific number of episodes
python scripts/run_rl.py --episodes 500
```

Once running:
- The snake will start moving **randomly** (this is normal!)
- Watch the **ε (epsilon)** value in the sidebar - it shows exploration rate
- As epsilon decreases, the snake gets **smarter**
- Score will improve over time as the agent learns

---

## 🎮 Your First Training Session

**What you'll see:**

1. **First 50 episodes**: Snake moves randomly, crashes a lot, score usually 0-1
   - This is **exploration** - the agent is discovering the environment
   - Q-values are being built up (watch Q-table size grow!)

2. **Episodes 50-200**: Snake starts moving toward food sometimes
   - **Exploitation** begins (using learned knowledge)
   - Average score improves to 2-4

3. **Episodes 200-1000**: Snake plays increasingly well
   - Q-values become more accurate
   - Average score reaches 5-10+

**Don't worry if it looks stupid at first - that's how learning works!**

---

## 🎛️ Interactive Controls (While Running)

### Keyboard Shortcuts
- **SPACE**: Pause training (snake stops)
- **S**: Save your trained model to `models/q_table.json`
- **L**: Load a previously saved model
- **T**: Toggle between Training and Inference mode
  - **Training**: Agent learns (Q-values update, epsilon active)
  - **Inference**: Agent just plays (no learning, epsilon=0)

### UI Sliders (Right Panel)
1. Adjust any slider
2. Click **"Apply"** button
3. Changes take effect immediately!

**What each slider does:**
- **Learning Rate (α)**: How fast Q-values change (0.001 to 0.1)
- **Discount (γ)**: How much agent values future rewards (0.8 to 0.99)
- **Epsilon Start**: Initial exploration rate (0.5 to 1.0)
- **Epsilon Decay**: How fast exploration decreases (0.95 to 0.999)
- **Speed**: Training speed in steps/second (10 to 200)

### Preset Buttons
- **Default**: Balanced settings (speed 50)
- **Fast**: Quick training (speed 150)
- **Slow**: Watch learning carefully (speed 20)
- **Turbo**: Maximum speed training (speed 200)

---

## 🧪 Easy Experiments (No Coding!)

### Experiment 1: Fast vs Slow Learning

**Step 1: Try Fast Learning**
```bash
python scripts/run_rl.py --preset fast_learning --episodes 200
```
Watch how quickly the agent improves!

**Step 2: Try Slow, Careful Learning**
```bash
python scripts/run_rl.py --preset slow_careful --episodes 200
```
Watch more carefully as the agent learns.

**Compare:** Which reaches higher average score? Which is more stable?

---

### Experiment 2: Save and Show Off Your Trained Agent

**Step 1: Train a model**
```bash
python scripts/run_rl.py --preset turbo --episodes 1000
```
Wait 5 minutes for training to complete. Press **S** to save when done.

**Step 2: Watch your trained agent play**
```bash
python scripts/run_rl.py --load --preset slow_careful
```
Press **T** to enter inference mode. Now watch it play without learning!

---

### Experiment 3: What Happens If...?

**While training is running:**

1. **Click "Slow" preset** → Training slows down so you can see each move
2. **Press SPACE** → Pause and examine current Q-values
3. **Press T** → Switch to inference mode (no more learning, just watch)
4. **Click "Turbo" preset** → Speed up again!

---

## 🎯 Understanding What You See

### The Numbers Explained

**ε (epsilon) = 0.850**
- Agent will explore (random action) 85% of the time
- Agent will exploit (use best known action) 15% of the time
- This number **decreases** as training progresses

**α (alpha) = 0.010**
- Q-values change by 1% of the prediction error
- Higher = faster learning but less stable
- Lower = slower learning but more stable

**Q-table size = 1,247**
- The agent has encountered 1,247 unique situations
- Each stores Q-values for 3 actions
- Grows as agent explores more states

**Q-Values: Straight: 2.45, Right: -1.20, Left: 3.67**
- The agent learned that turning **left** is best here (3.67, shown in green)
- Going straight is okay (2.45)
- Turning right is bad (-1.20, probably leads to danger)

---

## 🔬 Guided Experiments (With Simple Code Changes)

### Experiment 4: Make Agent More Greedy

**Edit `scripts/run_rl.py` and add before `main()`:**

```python
from modules.reinforcement_learning.config import config

# Make agent exploit more, explore less
config.EPSILON_START = 0.5    # Start with only 50% exploration
config.EPSILON_END = 0.01
config.EPSILON_DECAY = 0.98   # Decay faster
```

**Run it.** Does it learn faster or slower? Why?

---

### Experiment 5: Reward Engineering

**Add to `scripts/run_rl.py`:**

```python
# Version A: Big rewards
config.REWARD_FOOD = 50.0
config.REWARD_CLOSER_TO_FOOD = 5.0

# Or Version B: More punishment
config.REWARD_DEATH = -50.0
config.REWARD_FARTHER_FROM_FOOD = -5.0
```

**Try both.** Which helps the agent learn better?

---

### Experiment 6: Different Discount Factors

**Add to `scripts/run_rl.py`:**

```python
# Short-sighted agent (cares about immediate rewards)
config.DISCOUNT_FACTOR = 0.5

# Or far-sighted agent (cares about future rewards)
config.DISCOUNT_FACTOR = 0.99
```

**Question:** Which performs better in Snake? Why?

---

## 📊 Measuring Success

### Good Signs
- ✅ Epsilon decreasing (1.0 → 0.01)
- ✅ Average score increasing (0 → 5+)
- ✅ Q-table size growing (0 → 1000+)
- ✅ Best action (green) makes sense

### Bad Signs
- ❌ Average score stuck at 0-1 after 500 episodes
- ❌ Q-values all near zero
- ❌ Epsilon not decaying
- ❌ Agent behavior looks completely random

**If learning isn't working:**
1. Check epsilon is set to decay
2. Try higher learning rate (0.05)
3. Make sure rewards are configured
4. Use a preset to start fresh

---

## 🏆 Challenges

### Challenge 1: Train a Score-10 Agent
Can you train an agent that averages 10+ food per episode?
- Experiment with hyperparameters
- Try different reward structures
- Save your best model!

### Challenge 2: Fastest Learner
Can you get to average score of 5 in under 300 episodes?
- Hint: Adjust learning rate and epsilon decay
- Document your settings

### Challenge 3: Stable Training
Train for 1000 episodes where the last 100 episodes all score 5+.
- Focus on stability, not just peak performance
- Slower epsilon decay might help

---

## 💾 Save & Load Models

### Saving Your Progress

**Method 1: Keyboard shortcut**
- While training, press **S**
- Saves to `models/q_table.json`

**Method 2: Automatic save**
- Best model auto-saved as `models/q_table_best.json`
- Periodic saves every 100 episodes

### Loading a Saved Model

**Method 1: Command line**
```bash
python scripts/run_rl.py --load
```

**Method 2: While running**
- Press **L** to load from disk

**What's saved:**
- Q-table (all learned state-action values)
- Epsilon value (current exploration rate)
- Episode count (how much training)
- Statistics (for analysis)

---

## 🌐 Web Version

Try it in your browser (no installation needed!):
```
https://chinwh2019.github.io/intro-ai/reinforcement_learning/
```

**Differences from desktop:**
- ❌ Cannot save/load models (browser limitation)
- ❌ No learning curve plots (no matplotlib)
- ✅ Everything else works the same!
- ✅ Perfect for quick demos in class

---

## 💡 Tips for Success

1. **Start with a preset** - Don't guess parameters
2. **Use Turbo for training** - Fast iteration, then switch to Slow to watch
3. **Save often** - Press S every 100 episodes
4. **Compare runs** - Try different settings, keep notes
5. **Be patient** - Good learning takes 500+ episodes
6. **Watch epsilon** - Should go from 1.0 → 0.01
7. **Use inference mode** - After training, press T to watch agent play perfectly

---

## ❓ Common Questions

**Q: Why is the snake so bad at first?**
A: High epsilon means it's exploring randomly (not stupid, just learning!). This is necessary to discover the environment.

**Q: How long until it gets good?**
A: Usually 200-500 episodes. Use `--preset turbo` to train faster.

**Q: What if I want to train longer?**
A: `python scripts/run_rl.py --episodes 2000` (or any number)

**Q: Can I start where I left off?**
A: Yes! Save with **S**, then run with `--load` flag next time.

**Q: What's a good learning rate?**
A: Start with 0.01. If learning is too slow, try 0.05. If unstable, try 0.001.

**Q: Should I use reward shaping?**
A: It can help! The default gives small rewards for moving toward food. Experiment!

**Q: Why does average score go down sometimes?**
A: The agent is exploring less-optimal paths to learn about them. This is normal and healthy!

**Q: Can I compete with classmates?**
A: Yes! See who can train the best agent (highest average score over last 100 episodes).

---

## 📚 What You're Learning

**Core Concepts:**
- ✅ How agents learn from trial and error
- ✅ Balancing exploration (trying new things) vs exploitation (using what you know)
- ✅ Temporal difference learning (learning from prediction errors)
- ✅ Value functions (estimating how good states/actions are)
- ✅ Policy (strategy for choosing actions)

**Skills Developed:**
- ✅ Hyperparameter tuning
- ✅ Analyzing learning curves
- ✅ Understanding convergence
- ✅ Debugging ML systems
- ✅ Experiment design

**Real-World Applications:**
- 🎮 Game playing (chess, Go, video games)
- 🤖 Robot control (navigation, manipulation)
- 💰 Trading strategies (stock market)
- 🚗 Autonomous driving (decision making)
- 🏥 Treatment planning (medical decisions)

---

**Happy Learning! 🐍🎓**

*Remember: The agent starts dumb and gets smart - just like learning anything new!*
