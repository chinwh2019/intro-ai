# Reinforcement Learning - Overview

Reinforcement Learning (RL) is how agents learn to make good decisions through trial-and-error interaction with an environment.

## The Big Idea

**Learning by doing** - like teaching a dog tricks:
- Dog tries action (sit, jump, roll)
- Gets reward (treat or nothing)
- Learns which actions lead to treats
- Improves behavior over time

**No one tells the dog HOW to get treats - it figures it out through experience!**

## RL vs Other AI Approaches

### RL vs Supervised Learning

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|-------------------|----------------------|
| **Training data** | Labeled examples | Rewards from environment |
| **Feedback** | "This is correct answer" | "This got +10 reward" |
| **Learning** | Mimic examples | Discover through trial |
| **Example** | Image classification | Game playing |

### RL vs MDPs

| Aspect | MDP | Reinforcement Learning |
|--------|-----|----------------------|
| **Transitions** | Known | Unknown |
| **Rewards** | Known | Unknown initially |
| **Method** | Planning (math) | Learning (experience) |
| **Output** | Optimal policy | Learned policy |
| **Speed** | Fast (if small) | Slow (needs episodes) |

**RL is for when you DON'T KNOW the rules of the game!**

## The RL Framework

### Agent-Environment Interaction Loop

```
   ┌─────────┐
   │  Agent  │
   └────┬────┘
        │ action (a_t)
        ↓
   ┌─────────┐
   │  Env    │
   └────┬────┘
        │ state (s_t+1), reward (r_t+1)
        ↓
   ┌─────────┐
   │  Agent  │
   └─────────┘
```

**Step by step:**
1. Agent observes state s
2. Agent chooses action a
3. Environment transitions to state s'
4. Environment gives reward r
5. Agent updates its knowledge
6. Repeat!

## Q-Learning: The Algorithm We Use

### What is Q-Learning?

**Q(s, a)** = expected total reward from state s, taking action a, then acting optimally

**Goal:** Learn Q-values through experience, then use them to act optimally

**Key insight:** If you know Q(s,a) for all s and a, optimal policy is:
```python
π(s) = argmax_a Q(s,a)  # Choose action with highest Q
```

### The Q-Learning Update Rule

Every time you take action, update Q:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Components:**
- **Q(s,a)**: Current Q-value estimate
- **α**: Learning rate (how fast to learn, 0.01 typical)
- **r**: Reward just received
- **γ**: Discount factor (0.95 typical)
- **max Q(s',a')**: Best Q-value in next state
- **[r + γ max Q(s',a') - Q(s,a)]**: Prediction error (TD error)

**Intuition:** Move Q-value toward better estimate based on what you experienced

## The Snake Environment

### State Representation (11 features)

What the snake "sees":

**Danger Detection (3 features):**
1. Danger straight ahead? (True/False)
2. Danger if turn right? (True/False)
3. Danger if turn left? (True/False)

**Current Direction (4 features, one-hot):**
4. Moving up? (True/False)
5. Moving right? (True/False)
6. Moving down? (True/False)
7. Moving left? (True/False)

**Food Location (4 features):**
8. Food to the left? (True/False)
9. Food to the right? (True/False)
10. Food above? (True/False)
11. Food below? (True/False)

**Example state:**
```python
[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
# Danger straight, moving right, food left and below
```

### Action Space (3 actions)

**Relative to current direction:**
0. Continue straight
1. Turn right (90° clockwise)
2. Turn left (90° counter-clockwise)

**Why relative?** Simpler than absolute directions, easier to learn

### Rewards

**Designed to encourage good behavior:**
- **+10**: Ate food (primary goal!)
- **-10**: Crashed (wall or self-collision)
- **+1**: Moved closer to food (reward shaping)
- **-1.5**: Moved away from food (discourage)
- **-0.1**: Idle movement (encourage action)

**Reward shaping helps:** Agent learns faster by getting intermediate feedback

## Exploration vs Exploitation

### The ε-Greedy Strategy

**On each step, agent chooses:**
- With probability ε: Random action (explore)
- With probability 1-ε: Best known action (exploit)

**Epsilon (ε) starts high, decays over time:**
```
Episode 1:    ε = 1.0   (100% exploration - fully random)
Episode 100:  ε = 0.61  (61% exploration)
Episode 500:  ε = 0.08  (8% exploration)
Episode 1000: ε = 0.01  (1% exploration - mostly optimal)
```

**Why this works:**
- Early: Explore to discover environment
- Middle: Balance exploration and exploitation
- Late: Mostly exploit learned knowledge

**Decay formula:**
```
ε = ε * decay_rate  (each episode)
# Example: 1.0 * 0.995 = 0.995, then 0.995 * 0.995 = 0.990, ...
```

## Watching Q-Learning Learn

### Run the Module

```bash
python scripts/run_rl.py --preset turbo
```

### What to Observe

**Episodes 1-50: Random Chaos**
- Snake moves randomly (high ε)
- Crashes into walls
- Scores 0-1 mostly
- Q-table growing (discovering states)

**Episodes 50-200: Emerging Intelligence**
- ε decreasing (less random)
- Sometimes goes toward food
- Scores 2-4 average
- Q-values becoming meaningful

**Episodes 200-500: Competent Agent**
- Mostly exploiting (low ε)
- Usually goes toward food
- Scores 5-10 average
- Q-table well-developed

**Episodes 500-1000: Expert Agent**
- Minimal exploration
- Consistently finds food
- Scores 8-15 average
- Q-values stable

### The Q-Values Panel

**Example display:**
```
Q-Values:
Straight: 2.45
Right:   -1.20
Left:     3.67  ← (shown in green - best action)
```

**Interpretation:**
- Agent learned that turning LEFT is best here
- Expects +3.67 total reward from turning left
- Going straight is okay (+2.45)
- Turning right is bad (-1.20, probably hits wall)

**Watch Q-values:** They start at 0, become more accurate over time!

## Hyperparameters Explained

### Learning Rate (α)

**Controls how fast Q-values update:**
- α = 0.001: Very slow, very stable
- α = 0.01: Balanced (default)
- α = 0.1: Fast, but unstable

**Too high:** Noisy learning, oscillation
**Too low:** Slow convergence
**Just right:** Smooth improvement

### Discount Factor (γ)

**How much agent values future rewards:**
- γ = 0.5: Short-sighted (only cares about next reward)
- γ = 0.9: Balanced (default)
- γ = 0.99: Far-sighted (values distant future)

**In Snake:**
- Low γ: Agent might grab nearby food, ignore long-term strategy
- High γ: Agent plans ahead, avoids traps

### Epsilon Decay Rate

**How fast exploration decreases:**
- 0.999: Very slow decay (explore longer)
- 0.995: Moderate (default)
- 0.99: Fast decay (exploit sooner)

**Too fast:** Agent settles on suboptimal policy (didn't explore enough)
**Too slow:** Wastes time exploring when should be exploiting

## Training Phases

### Phase 1: Pure Exploration (Episodes 0-100)
- **Goal:** Discover the environment
- **ε**: 1.0 → 0.6
- **Behavior:** Mostly random
- **Q-table:** Growing rapidly
- **Performance:** Poor (score 0-2)

### Phase 2: Learning Acceleration (Episodes 100-300)
- **Goal:** Learn what works
- **ε**: 0.6 → 0.2
- **Behavior:** Mix of random and smart
- **Q-table:** Values becoming accurate
- **Performance:** Improving (score 2-5)

### Phase 3: Refinement (Episodes 300-1000)
- **Goal:** Perfect the policy
- **ε**: 0.2 → 0.01
- **Behavior:** Mostly smart moves
- **Q-table:** Fine-tuning values
- **Performance:** Good (score 5-12)

## Success Metrics

### What to Track

**Short-term (per episode):**
- Score (food eaten)
- Total reward
- Steps survived

**Long-term (over 100 episodes):**
- **Average score** (most important!)
- Max score achieved
- Average reward
- Q-table size (unique states)

### Good Learning Looks Like

**Average score curve:**
```
Score
  12 |                            ___
  10 |                      ___/
   8 |                ___/
   6 |          ___/
   4 |    ___/
   2 |___/
   0 |_________________________________
      0   200   400   600   800  1000
                Episodes
```

Smooth upward trend = healthy learning!

### Bad Learning Looks Like

**Flat line:**
```
Score
   2 |_________________________________
   0 |_________________________________
      0   200   400   600   800  1000
```
**Problem:** Hyperparameters wrong, or not enough exploration

**Oscillation:**
```
Score
  10 |  /\    /\    /\
   5 | /  \  /  \  /  \
   0 |/    \/    \/    \
      0   200   400   600
```
**Problem:** Learning rate too high

## Interactive Training

### Using the Parameter Panel

**While training:**
1. **Adjust learning rate slider** → Change how fast agent learns
2. **Adjust epsilon decay slider** → Change exploration schedule
3. **Adjust speed slider** → Make training faster/slower
4. **Click Apply** → Changes take effect immediately!

**Experiment live:**
- Start with default
- If learning too slow → increase learning rate
- If unstable → decrease learning rate
- If getting stuck → slow down epsilon decay

### Training vs Inference Mode

**Training Mode (default):**
- Agent learns from every step
- Q-values update continuously
- ε > 0 (some exploration)
- Episode counter increases

**Inference Mode (press T):**
- Agent doesn't learn (Q-values frozen)
- ε = 0 (no exploration, pure exploitation)
- Just watch trained agent play
- Demo counter increases

**Use case:** Train first, then press T to watch your trained agent perform!

## Understanding the Statistics

### Sidebar Panel Explained

**Episode: 347/1000**
- Currently on episode 347
- Training for 1000 total episodes

**Score: 6**
- Ate 6 food items this episode
- Higher is better!

**Steps: 245**
- Survived 245 steps this episode
- Longer = better (unless stuck)

**ε (epsilon): 0.183**
- Will explore 18.3% of the time
- Will exploit 81.7% of the time

**α (alpha): 0.010**
- Learning rate
- Q-values change by 1% of prediction error

**Q-table size: 1,847**
- Encountered 1,847 unique states
- Each stores 3 Q-values (one per action)

**Recent Performance (Last 100 episodes):**
- **Avg Score: 5.2** - Main success metric!
- **Max Score: 12** - Best episode
- **Avg Reward: 38.4** - Total reward per episode

## Common Pitfalls

### Pitfall 1: Impatience
**Symptom:** "Why is snake so stupid after 50 episodes?"
**Reality:** RL needs 200-500 episodes to show real progress
**Solution:** Use --preset turbo, be patient

### Pitfall 2: No Exploration
**Symptom:** Agent gets stuck in local optimum
**Reality:** ε decayed too fast, didn't explore enough
**Solution:** Slower epsilon decay (0.999 instead of 0.995)

### Pitfall 3: Too Much Learning Rate
**Symptom:** Performance oscillates wildly
**Reality:** α too high, Q-values jumping around
**Solution:** Lower learning rate (0.001 instead of 0.01)

### Pitfall 4: Wrong Rewards
**Symptom:** Agent learns weird behavior
**Reality:** Reward structure encourages wrong thing
**Solution:** Check reward values make sense

## Real-World RL Applications

### Where Q-Learning is Used

**Game Playing:**
- 🎮 Atari games (DeepMind DQN)
- 🀄 Backgammon (TD-Gammon, 1992)
- 🎯 Board games

**Robotics:**
- 🤖 Robot walking/running
- 🦾 Manipulation tasks
- 🚁 Drone control

**Resource Management:**
- 🌡️ Data center cooling (Google)
- 💰 Trading strategies
- 📊 Ad placement

**Optimization:**
- 🚦 Traffic light timing
- 📡 Network routing
- 🏭 Manufacturing scheduling

## Learning Path

### Beginner
1. [What is Reinforcement Learning?](What-is-RL)
2. [Q-Learning Explained Simply](Q-Learning-Simple)
3. [Understanding Epsilon-Greedy](Epsilon-Greedy-Explained)

### Intermediate
4. [Temporal Difference Learning](TD-Learning)
5. [Reward Shaping Guide](Reward-Shaping)
6. [Hyperparameter Tuning](RL-Hyperparameters)

### Advanced
7. [SARSA vs Q-Learning](SARSA-vs-Q-Learning)
8. [Deep Q-Networks (DQN)](Deep-Q-Networks)
9. [Policy Gradient Methods](Policy-Gradients)

## Key RL Concepts

### 1. Return (Total Reward)

**Undiscounted return:**
```
G_t = r_t + r_t+1 + r_t+2 + ... + r_T
```

**Discounted return (more common):**
```
G_t = r_t + γr_t+1 + γ²r_t+2 + ... + γ^(T-t)r_T
```

**Why discount?**
- Uncertainty about distant future
- Sooner rewards more valuable
- Mathematical convergence

### 2. Value Function

**V(s):** Expected return starting from state s
```
V(s) = E[G_t | s_t = s]
```

**In Q-learning:** V(s) = max_a Q(s,a)

### 3. Q-Function (Action-Value Function)

**Q(s,a):** Expected return from state s, taking action a
```
Q(s,a) = E[G_t | s_t = s, a_t = a]
```

**This is what we learn!** Q-table stores Q(s,a) for all seen (s,a) pairs.

### 4. Policy

**π(s):** Which action to take in state s

**Deterministic policy:**
```python
π(s) = argmax_a Q(s,a)  # Always choose best action
```

**Stochastic policy (ε-greedy):**
```python
if random() < ε:
    a = random_action()  # Explore
else:
    a = argmax_a Q(s,a)  # Exploit
```

### 5. Temporal Difference (TD) Error

**The learning signal:**
```
δ_t = r + γ max Q(s',a') - Q(s,a)
```

**Interpretation:**
- If δ > 0: You got more reward than expected → increase Q
- If δ < 0: You got less reward than expected → decrease Q
- If δ = 0: Perfect prediction → no change

**Q-learning update:**
```
Q(s,a) ← Q(s,a) + α * δ_t
```

## Watching Q-Learning Learn

### Run the Module

```bash
python scripts/run_rl.py
```

### Understanding What You See

**Game Area (Left):**
- Green blocks: Snake (head is brighter)
- Red block: Food
- The snake will initially move randomly!

**Stats Panel (Right):**
- **Q-Values**: Real-time learning
- **ε**: Exploration rate (watch it decrease!)
- **Average Score**: Main success metric

**Don't worry when:**
- Snake looks stupid initially (it's exploring!)
- Score is 0 for first 50 episodes
- It crashes a lot at first

**Good signs:**
- ε decreasing (1.0 → 0.01)
- Average score increasing
- Q-table size growing
- Snake sometimes goes toward food

## The Learning Curve

### Typical Training Progress

**Episodes 0-100: Discovery Phase**
```
ε: 1.0 → 0.6
Avg Score: 0.5
Behavior: Random wandering
Q-table: 200 states
```

**Episodes 100-300: Learning Phase**
```
ε: 0.6 → 0.2
Avg Score: 3.2
Behavior: Mix of exploration and smart moves
Q-table: 800 states
```

**Episodes 300-700: Improvement Phase**
```
ε: 0.2 → 0.04
Avg Score: 6.8
Behavior: Mostly smart, occasional exploration
Q-table: 1,500 states
```

**Episodes 700-1000: Mastery Phase**
```
ε: 0.04 → 0.01
Avg Score: 9.5
Behavior: Consistently good play
Q-table: 2,000 states
```

## Experiments for Students

### Experiment 1: Learning Rate Effects

**Fast learning (α = 0.05):**
```bash
python scripts/run_rl.py --lr 0.05 --episodes 300
```
Converges quickly, but might be noisy

**Slow learning (α = 0.001):**
```bash
python scripts/run_rl.py --lr 0.001 --episodes 300
```
Slower convergence, but smoother

**Compare:** Which reaches higher average score?

### Experiment 2: Exploration Schedules

**Fast decay (greedy quickly):**
```bash
python scripts/run_rl.py --preset greedy --episodes 500
```

**Slow decay (explore longer):**
```bash
python scripts/run_rl.py --preset slow_careful --episodes 500
```

**Compare:** Which finds better final policy?

### Experiment 3: Reward Shaping

Edit `modules/reinforcement_learning/config.py`:

**Version A: Only terminal rewards**
```python
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_CLOSER_TO_FOOD = 0.0  # No shaping
REWARD_FARTHER_FROM_FOOD = 0.0
```

**Version B: Strong shaping**
```python
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_CLOSER_TO_FOOD = 5.0  # Strong shaping
REWARD_FARTHER_FROM_FOOD = -5.0
```

**Compare:** Which learns faster? Why?

## The Q-Table

### What's Stored

**Sparse dictionary:**
```python
{
  (1,0,0,0,1,0,0,1,0,0,1): [2.3, -1.1, 3.8],  # State → [Q(s,straight), Q(s,right), Q(s,left)]
  (0,1,0,0,0,1,0,0,1,1,0): [5.2, 1.4, -0.8],
  # ... thousands more entries
}
```

**Grows during training:**
- Episode 1: 0 states
- Episode 100: ~500 states
- Episode 1000: ~2000 states

**Memory efficient:**
- Only stores visited states
- Continuous state space → infinite possible states
- Only ~2000 actually encountered!

### Analyzing a Trained Q-Table

**Load saved model:**
```bash
python scripts/run_rl.py --load
```

**Check statistics:**
- Q-table size: How much learned
- Episodes trained: How much experience
- Average Q-value: Overall optimism/pessimism

## Training Tips

### Getting Good Results

1. **Start with presets** - Don't guess hyperparameters
2. **Use turbo for training** - Fast iteration
3. **Train for 1000+ episodes** - Don't stop early
4. **Save good models** - Press S when avg score high
5. **Use inference mode** - Press T to watch without learning

### Debugging Poor Performance

**If average score stuck at 0-1:**
- Check ε is decaying
- Try higher learning rate
- Verify rewards are configured
- Train longer (500+ episodes)

**If performance oscillates:**
- Lower learning rate
- Slower epsilon decay
- Check reward shaping isn't too strong

**If agent does one thing repeatedly:**
- Didn't explore enough
- Reset and use slower ε decay

## Off-Policy vs On-Policy

### Q-Learning is Off-Policy

**What this means:**
- Agent follows ε-greedy policy (sometimes random)
- But learns about greedy policy (always best)
- Learns optimal Q* while behaving non-optimally

**Advantage:** Can learn from any experience (even random exploration)

### SARSA is On-Policy (Comparison)

**What this means:**
- Agent follows ε-greedy policy
- Learns about that same ε-greedy policy
- More conservative in stochastic environments

**Difference:** Q-learning more aggressive, SARSA safer

## When to Use Q-Learning

**Use Q-Learning when:**
- ✅ Discrete state and action spaces (or can discretize)
- ✅ Don't know environment model
- ✅ Can simulate many episodes
- ✅ Want optimal policy
- ✅ Can afford exploration

**Don't use Q-Learning when:**
- ❌ Continuous actions (use policy gradient)
- ❌ Huge state space (use function approximation)
- ❌ Can't afford many episodes (use model-based RL)
- ❌ Safety critical (no exploration allowed)

## From Q-Learning to Deep RL

**Q-Learning limitations:**
- Q-table grows with state space
- Can't handle millions of states
- No generalization

**Solution: Deep Q-Networks (DQN)**
- Replace Q-table with neural network
- Input: state → Output: Q-values
- Handles huge state spaces (pixels!)
- Used for Atari, Go, robotic control

**Your Q-Learning foundation** prepares you for understanding DQN!

## Summary

**Q-Learning is:**
- Model-free (doesn't need T and R)
- Off-policy (learns optimal while exploring)
- Guaranteed to converge (under conditions)
- Simple to implement
- Foundation of modern Deep RL

**Key to success:**
1. Good state representation
2. Reasonable reward structure
3. Sufficient exploration
4. Enough training episodes
5. Proper hyperparameters

---

**Next:** [Q-Learning Deep Dive](Q-Learning-Deep-Dive) | [Reward Shaping](Reward-Shaping)

**Try it:** `python scripts/run_rl.py --preset turbo`
