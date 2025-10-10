# Markov Decision Processes (MDPs) - Overview

MDPs are mathematical frameworks for modeling decision-making when outcomes are partly random and partly under your control.

## The Big Question

**"What should I do in each situation to maximize my total reward over time?"**

That's what MDPs solve!

## Real-World Example: Self-Driving Car

**States:** Car positions and speeds
**Actions:** Accelerate, brake, turn
**Transitions:** Partly predictable (physics), partly random (other drivers)
**Rewards:** Reach destination (+100), crash (-1000), time penalty (-1 per second)
**Goal:** Find policy that maximizes expected reward

## MDP Components

An MDP is defined by 5 things:

### 1. States (S)

All possible situations you could be in.

**In our grid world:**
- Each cell is a state
- 5×5 grid = 25 states
- State example: (row=2, col=3)

**In real world:**
- Robot: (x, y, battery_level, carrying_object?)
- Game: (player_position, enemy_positions, inventory)

### 2. Actions (A)

Things you can do in each state.

**In our grid world:**
- UP, DOWN, LEFT, RIGHT
- Same actions available in all non-terminal states

**In real world:**
- Robot: move, pick_up, charge, drop
- Game: attack, defend, run, use_item

### 3. Transitions (T)

**T(s, a, s')** = probability of reaching s' from s by doing a

**Key insight:** Outcomes are **stochastic** (probabilistic)!

**Example in our grid (with 20% noise):**
```python
You're at (2,2), choose action UP:
- 80% chance: move to (1,2)  [intended]
- 10% chance: move to (2,1)  [slip left]
- 10% chance: move to (2,3)  [slip right]
```

**This models:** Slippery floors, robot actuator errors, uncertain environments

### 4. Rewards (R)

**R(s, a)** or **R(s)** = immediate reward for being in state/taking action

**In our grid world:**
- Goal state: +1.0 (yay!)
- Danger state: -1.0 (ouch!)
- Every other step: -0.04 (living cost)

**Living cost encourages:** Finding goal quickly, not wandering

### 5. Discount Factor (γ)

**γ** (gamma) = how much you value future rewards (0 to 1)

**Examples:**
- γ = 0.9: Future reward worth 90% of immediate reward
- γ = 0.5: Future reward worth 50% of immediate reward
- γ = 0.99: Future reward worth 99% of immediate reward

**Effect:**
- γ near 0: Short-sighted (greedy)
- γ near 1: Far-sighted (patient)

## What MDPs Compute

### Value Function V(s)

**V(s)** = expected total discounted reward starting from state s, acting optimally

**Example:**
```
V(state_near_goal) = 0.85  (good state!)
V(state_near_danger) = -0.42  (bad state!)
V(obstacle) = undefined  (can't enter)
```

**Visualization:** Green cells = high value, red cells = low value

### Q-Function Q(s,a)

**Q(s,a)** = expected total reward from state s, taking action a, then acting optimally

**Example:**
```
At state (2,2):
Q((2,2), UP) = 0.65
Q((2,2), RIGHT) = 0.81  ← Best action!
Q((2,2), DOWN) = 0.23
Q((2,2), LEFT) = 0.45
```

**Visualization:** Toggle Q-values display (Q key) to see all Q(s,a) values

### Policy π(s)

**π(s)** = best action to take in state s

**Derived from Q-values:**
```python
π(s) = argmax_a Q(s,a)  # Action with highest Q-value
```

**Visualization:** White arrows showing optimal action in each cell

## The Bellman Equation

### Bellman Optimality Equation

The foundation of MDPs:

```
V*(s) = max_a [ R(s,a) + γ Σ T(s,a,s') V*(s') ]
```

**In English:**
1. For each possible action a
2. Calculate: immediate reward + discounted future value
3. Pick the action that maximizes this
4. That's your optimal value!

**Recursive definition:** Value of state depends on values of next states

### Value Iteration Algorithm

**Iteratively apply Bellman equation:**

```
1. Initialize all V(s) = 0
2. Repeat until convergence:
   For each state s:
     V_new(s) = max_a [ R(s,a) + γ Σ T(s,a,s') V_old(s') ]
3. Extract policy: π(s) = argmax_a Q(s,a)
```

**Watch it happen:**
```bash
python scripts/run_mdp.py
# Press SPACE, watch values spread from goal/danger
```

## Understanding Stochasticity

### Deterministic (Noise = 0%)

```
You choose RIGHT:
→ Always move right
→ Predictable outcome
```

**Policy:** Can be aggressive (go straight to goal)

### Stochastic (Noise = 20%)

```
You choose RIGHT:
→ 80% move right
→ 10% move up
→ 10% move down
```

**Policy:** Might be more conservative (avoid danger)

### Experiment

```bash
# No noise
python scripts/run_mdp.py --noise 0.0
# Note policy pattern

# High noise
python scripts/run_mdp.py --noise 0.4
# Note policy pattern - more cautious?
```

## Visualizing Value Iteration

### What You See

**Iteration 0:**
```
All cells gray (value = 0)
```

**Iteration 1:**
```
Goal: green (value ≈ +1.0)
Goal neighbors: light green (value ≈ +0.8)
Danger: red (value ≈ -1.0)
Danger neighbors: light red (value ≈ -0.8)
```

**Iteration 5:**
```
Green spreading from goal
Red spreading from danger
Values getting more accurate
```

**Iteration 20:**
```
Values stabilized
Policy arrows stable
Convergence reached!
```

## The Controls Explained

### SPACE - Start/Pause Value Iteration
- Watch values update in real-time
- Each press of space = run all iterations to convergence

### S - Step (When Paused)
- Advance one iteration at a time
- See exactly how values change
- Great for understanding propagation

### V - Toggle Value Display
- Show/hide numerical values in cells
- ON: See exact V(s) values
- OFF: Just see color gradients

### P - Toggle Policy Display
- Show/hide white arrows
- Arrows = optimal action in each state
- Follow arrows = optimal path

### Q - Toggle Q-Values
- Show all Q(s,a) values
- See value of each action in each state
- Understand why policy chooses that action

### R - Reset Grid
- Generate new random layout
- New obstacle positions
- New start position
- Goal/danger positions stay same

## Key Insights

### 1. Planning vs Learning

**MDP (Planning):**
- You KNOW transition probabilities
- You KNOW rewards
- You COMPUTE optimal policy
- No trial-and-error needed

**RL (Learning):**
- You DON'T KNOW transitions
- You DON'T KNOW rewards initially
- You LEARN by trying things
- Trial-and-error required

### 2. Optimal Policy Exists

**Bellman equation guarantees:**
- Optimal policy exists
- Value iteration finds it
- Policy is deterministic (usually)
- Same policy regardless of start state

### 3. Values Propagate

**Intuition:**
- Goal has value = immediate reward
- Neighbors get value = discounted goal value
- Values spread outward like waves
- Each iteration updates all states simultaneously

## Practical Applications

### 1. Game Playing
- Chess: State = board, Action = move
- MDP solves endgames (tablebases)
- Policy = best move in each position

### 2. Resource Management
- State = inventory levels
- Actions = produce, buy, sell
- Rewards = profit/loss
- Policy = optimal inventory strategy

### 3. Robot Control
- State = robot location + battery
- Actions = move, charge
- Transition noise = motor errors
- Policy = optimal navigation strategy

### 4. Finance
- State = portfolio + market conditions
- Actions = buy, sell, hold
- Stochastic = market randomness
- Policy = trading strategy

## Comparison with Search

| Aspect | Search (A*) | MDP (Value Iter) |
|--------|-------------|------------------|
| **Goal** | Find path | Find policy |
| **Output** | One path | Action for every state |
| **Uncertainty** | None | Stochastic transitions |
| **Rewards** | None (just goal) | Every state has reward |
| **Use when** | Deterministic | Stochastic |

**Both:** Navigate from start to goal optimally!

**Difference:** MDP handles uncertainty, gives complete strategy

## Learning Path

### Beginner
1. [MDP Fundamentals](MDP-Fundamentals) - Core concepts
2. [Bellman Equations Explained](Bellman-Equations) - The math
3. [Value Iteration Tutorial](Value-Iteration-Tutorial) - The algorithm

### Intermediate
4. [Policy Iteration](Policy-Iteration) - Alternative algorithm
5. [Discount Factor Effects](Discount-Factor-Effects) - Tuning γ
6. [Stochasticity Deep Dive](Stochasticity-Deep-Dive) - Handling uncertainty

### Advanced
7. [Policy Evaluation](Policy-Evaluation) - How good is a policy?
8. [Partially Observable MDPs](POMDPs) - Unknown state
9. [Continuous MDPs](Continuous-MDPs) - Infinite states

## Common Questions

**Q: Why is it called "Markov"?**
A: The Markov property: Future depends only on current state, not history. Past doesn't matter!

**Q: What if I don't know transition probabilities?**
A: Then you can't use MDP directly - use Reinforcement Learning instead!

**Q: How is this different from Q-Learning?**
A: MDP assumes you know T and R. Q-Learning learns them by trial-and-error.

**Q: Why use discount factor?**
A: Mathematically: ensures convergence. Practically: models time preference and uncertainty about distant future.

**Q: Can states be continuous?**
A: Yes! But then you need function approximation (advanced topic).

---

**Ready to dive in?** [MDP Fundamentals](MDP-Fundamentals) →
