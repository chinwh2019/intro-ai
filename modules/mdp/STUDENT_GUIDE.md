# MDP Module - Student Quick Guide

## 🚀 How to Run

```bash
# Basic - just run it!
python scripts/run_mdp.py

# With different settings
python scripts/run_mdp.py --preset deterministic  # No randomness
python scripts/run_mdp.py --preset high_noise     # More uncertainty
python scripts/run_mdp.py --grid-size 8           # Larger grid
```

Once running:
- Press **SPACE** to start value iteration
- Watch the **values** spread from goal/danger states
- Press **P** to see the **policy arrows** (optimal actions)
- Press **V** to toggle value display on/off

---

## 🎮 Your First Experience

**Step-by-step walkthrough:**

1. **Run the program**
   ```bash
   python scripts/run_mdp.py
   ```

2. **Observe the grid**
   - 🟡 **Gold cell**: Goal (reward +1.0) - try to reach this!
   - 🔴 **Red cell**: Danger (reward -1.0) - avoid this!
   - ⬜ **Gray cells**: Obstacles (can't move here)
   - 🔵 **Blue circle**: Starting position
   - All other cells start with value 0.0 (gray)

3. **Press SPACE to start value iteration**
   - Watch values **spread outward** from goal and danger
   - **Green cells**: Positive value (good states to be in)
   - **Red cells**: Negative value (bad states to be in)
   - Darker green = higher value

4. **Press P to see the policy**
   - **White arrows** show the best action in each state
   - Arrows point toward the goal (if safe)
   - Arrows avoid danger and obstacles
   - This is the **optimal strategy**!

---

## 🎛️ Controls

### Keyboard
- **SPACE**: Start/Pause value iteration
- **S**: Step through one iteration (when paused)
- **R**: Reset grid (generate new random layout)
- **V**: Toggle value numbers display
- **P**: Toggle policy arrows
- **Q**: Toggle Q-values display
- **ESC**: Quit

### What You're Watching

**Value Iteration:**
- Starts with all values = 0.0
- Each iteration updates every state's value
- Values propagate from goal/danger outward
- Converges when changes become tiny (<0.001)
- Usually takes 20-50 iterations

**The Policy:**
- Automatically extracted from values
- Shows best action in each state
- Updates as values improve
- Becomes optimal when values converge

---

## 🧪 Easy Experiments

### Experiment 1: Noise Effects

**No noise (deterministic):**
```bash
python scripts/run_mdp.py --preset deterministic
```
- Press SPACE, then P to see policy
- Notice policy goes **straight to goal**

**High noise (40% chance of slip):**
```bash
python scripts/run_mdp.py --preset high_noise
```
- Press SPACE, then P to see policy
- Notice policy might **avoid being near danger** more

**Question:** Why does noise make the agent more cautious?

---

### Experiment 2: Discount Factor (Time Preference)

**Short-sighted agent (discount = 0.5):**
```bash
python scripts/run_mdp.py --discount 0.5
```
- Agent only cares about immediate rewards
- Future rewards worth 50% of current

**Far-sighted agent (discount = 0.95):**
```bash
python scripts/run_mdp.py --discount 0.95
```
- Agent values future rewards highly
- Future rewards worth 95% of current

**Question:** How does this affect the policy? Which is better?

---

### Experiment 3: Living Reward

**Change the cost of living** by editing `config.py`:

```python
# Cheap living (agent takes longer routes)
config.LIVING_REWARD = -0.01

# Expensive living (agent rushes)
config.LIVING_REWARD = -0.2
```

**Question:** How does living cost affect path choice?

---

## 📊 Understanding the Numbers

### Sidebar Statistics

**Iteration: 23**
- Value iteration has run 23 times
- Each iteration updates all state values
- Usually converges in 20-50 iterations

**Discount (γ): 0.90**
- Future rewards are worth 90% of immediate rewards
- 0.0 = only care about now
- 1.0 = value all future rewards equally

**Noise: 0.20**
- 20% chance action doesn't work as intended
- 80% chance it moves where you want
- 10% chance it slips perpendicular left
- 10% chance it slips perpendicular right

**Living Reward: -0.04**
- You lose 0.04 reward for each step
- Encourages finding goal quickly
- Negative = penalty (cost of living)

---

## 🎨 What the Colors Mean

### Cell Colors (Values)
- **Bright Green**: High positive value (+0.5 to +1.0) - great states!
- **Light Green**: Moderate positive value (0.0 to +0.5) - okay states
- **Gray**: Zero value (0.0) - neutral
- **Light Red**: Moderate negative value (-0.5 to 0.0) - risky
- **Dark Red**: High negative value (-1.0 to -0.5) - dangerous!

### Special Cells
- **Gold with border**: Goal state (terminal, +1.0)
- **Red with border**: Danger state (terminal, -1.0)
- **Gray**: Obstacle (can't enter)
- **Blue circle**: Start position

### Policy Arrows
- **White arrows**: Point to best action
- **Arrow direction**: Which way to move
- **No arrow**: Terminal state or obstacle

---

## 🧠 Key Concepts Explained Simply

### What is an MDP?

Think of it like a **game with rules**:
- You're in a **state** (a position on the grid)
- You choose an **action** (up, down, left, right)
- The world **might not do exactly what you want** (noise/stochasticity)
- You get a **reward** (good or bad)
- You want to **maximize total rewards** over time

### What is Value Iteration?

It's a way to figure out **how good each position is**:
- "If I'm in this cell, what's the best total reward I can expect?"
- Calculates this for **every cell**
- Uses these values to decide the **best action**
- No need to try every possibility - math figures it out!

### What is a Policy?

Your **strategy** or **plan**:
- "If I'm in cell (2,3), I should go RIGHT"
- Specifies best action for every state
- The arrows show this strategy visually
- Following the policy = playing optimally

### What is Stochasticity?

**Uncertainty** in the environment:
- You choose "go right" but sometimes you slip "up" or "down"
- Real world is like this (robots slip, cars skid, etc.)
- MDP handles this mathematically
- Optimal policy accounts for possible slips

---

## 🎯 Learning Activities

### Beginner: Watch and Understand (15 min)

1. Run: `python scripts/run_mdp.py --preset deterministic`
2. Press **SPACE** to start value iteration
3. Press **S** to step slowly, one iteration at a time
4. Watch how values change each iteration
5. Press **P** to see the policy

**Questions to answer:**
- Which cells get positive values first?
- Which cells get negative values first?
- Why do values near the goal turn green quickly?
- What direction do most arrows point?

### Intermediate: Noise Comparison (20 min)

**Part 1: No noise**
```bash
python scripts/run_mdp.py --noise 0.0
```
- Note the policy pattern
- Screenshot or sketch it

**Part 2: High noise**
```bash
python scripts/run_mdp.py --noise 0.4
```
- Note the policy pattern
- Compare with Part 1

**Questions:**
- How does the policy change?
- Does it stay farther from danger?
- Why or why not?

### Advanced: Discount Exploration (30 min)

Try discount factors: **0.3, 0.7, 0.9, 0.99**

For each:
1. Run value iteration
2. Record: iterations to converge, policy pattern
3. Answer: How do values near the goal change?

**Write a short report explaining the effect of discount on:**
- Value magnitudes
- Policy aggressiveness
- Convergence speed

---

## 🔧 Customization Ideas

### Change Grid Layout

**Edit `modules/mdp/environments/grid_world.py`:**

```python
# Add more obstacles
obstacles = [(1,1), (1,2), (2,1), (3,3), (4,4)]

# Move goal/danger
goal_pos = (7, 7)
danger_pos = (3, 3)
```

### Create New Reward Structure

**Edit `config.py`:**

```python
# Add checkpoints
config.CHECKPOINT_REWARD = 0.5

# Add multiple goals
config.GOAL_REWARD = 10.0
config.SUBGOAL_REWARD = 3.0
```

Then modify the grid world to include these!

---

## 📚 Learning Outcomes

After using this module, you should be able to:

- ✅ Explain what an MDP is
- ✅ Describe the Bellman equation in plain English
- ✅ Understand value functions and policies
- ✅ Explain exploration vs exploitation
- ✅ Recognize when to use planning vs learning
- ✅ Tune discount factor for different scenarios
- ✅ Account for uncertainty in decision making

---

## 🆚 MDP vs Reinforcement Learning

**MDP (this module):**
- You **know the rules** (transition probabilities, rewards)
- You **plan** using math (value iteration)
- No trial-and-error needed
- Fast convergence (20-50 iterations)
- **Example:** Game where you know all rules

**RL (Snake Q-Learning module):**
- You **don't know the rules** initially
- You **learn** by trying things (Q-learning)
- Trial-and-error required
- Slow convergence (100s-1000s of episodes)
- **Example:** Real robot in unknown environment

**Both solve similar problems, different assumptions!**

---

## 🏅 Challenge Problems

### Challenge 1: Design a Grid World
Create a grid where:
- There are multiple paths to the goal
- One path is shorter but riskier (near danger)
- One path is longer but safer
- Does the policy choose safe or risky? Why?

### Challenge 2: Cliff World
Create a grid that looks like:
```
G . . . . . . . S
D D D D D D D D .
```
(G=Goal, S=Start, D=Danger, .=Safe)

What policy emerges? Does it hug the cliff or stay far?

### Challenge 3: Convergence Analysis
Record how many iterations it takes to converge for:
- Grid size: 5, 6, 7, 8
- Noise: 0.0, 0.2, 0.4
- Discount: 0.5, 0.9, 0.99

Plot the results. What patterns do you see?

---

## 💻 Running Examples

The module includes ready-to-run examples:

```bash
python modules/mdp/examples.py
```

Choose from:
1. Basic value iteration demo
2. Policy comparison (different discounts)
3. Noise effect visualization
4. Custom reward structure
5. Bellman equation explanation

---

**Good luck and have fun planning! 🎲🎯**

*Remember: MDPs are about making optimal decisions when you know the rules but outcomes are uncertain!*
