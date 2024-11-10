# Snake Q-Learning AI

A sophisticated implementation of Q-Learning applied to the classic Snake game, featuring real-time visualization of the learning process and interactive controls.

## Table of Contents
- [Snake Q-Learning AI](#snake-q-learning-ai)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Usage](#usage)
    - [Command Line Arguments](#command-line-arguments)
  - [Project Structure](#project-structure)
    - [Main Classes](#main-classes)
  - [Technical Details](#technical-details)
    - [State Representation](#state-representation)
    - [Action Space](#action-space)
    - [Reward System](#reward-system)
  - [Configuration](#configuration)
  - [Visualization](#visualization)
  - [Training Process](#training-process)
  - [File Management](#file-management)
    - [Saved Files](#saved-files)

## Overview

This project implements a Q-Learning agent that learns to play the Snake game. It features comprehensive visualization tools to help understand the learning process, making it suitable for both educational purposes and AI experimentation.

## Features

- **Q-Learning Implementation**: Advanced Q-Learning algorithm with experience replay
- **Real-time Visualization**: Dynamic display of:
  - Q-values
  - Exploration vs. Exploitation balance
  - Learning progress
  - Game state
  - Reward system
- **Interactive Controls**: Real-time adjustment of:
  - Learning rate
  - Epsilon (exploration rate)
  - Game speed
- **Training Management**:
  - Save/load functionality
  - Best model preservation
  - Training statistics tracking
- **Multiple Operation Modes**:
  - New training
  - Continue training
  - Watch trained agent
  - Interactive play

## Usage

Run the program with default settings:
```
python rl-snake.py
```

Custom training session:
```
python rl-snake.py --episodes 1000 --speed 50
```

Watch trained agent:
```
python rl-snake.py --episodes 5 --speed 30 --delay 0.2
```


### Command Line Arguments
- `--episodes`: Number of episodes to run (default: 1000)
- `--speed`: Game speed (default: 50)
- `--delay`: Delay between steps (default: 0.1)

## Project Structure

### Main Classes

1. **Config**: Configuration settings management
2. **SnakeGameAI**: Core game environment
3. **QLearningAgent**: Q-Learning implementation
4. **LearningVisualizer**: Real-time learning visualization
5. **ExperienceReplay**: Memory buffer for experiences
6. **TrainingStats**: Training metrics tracking
7. **InteractiveControls**: Runtime parameter adjustment

## Technical Details

### State Representation
The state space includes:
- Danger detection (straight, right, left)
- Current direction (one-hot encoded)
- Food location (relative position)
- Normalized distances to food

### Action Space
Three possible actions:
1. Continue straight
2. Turn right
3. Turn left

### Reward System
- Food collection: +20
- Game over: -10
- Moving closer to food: +1
- Moving away from food: -1
- No progress: -0.1

## Configuration

Key parameters in `Config` class:
```
LEARNING_RATE = 0.01
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 100000
```


## Visualization

The visualization panel includes:
1. **Current Game State**:
   - Snake position
   - Food location
   - Score
   
2. **Learning Metrics**:
   - Q-values for each action
   - Exploration/Exploitation ratio
   - Current reward
   - Total reward

3. **Training Progress**:
   - Episode number
   - Success rate
   - Average score

## Training Process

The training consists of three phases:
1. **Exploration Phase** (200 episodes):
   - High epsilon value
   - Random action selection
   
2. **Transition Phase** (500 episodes):
   - Gradually decreasing epsilon
   - Balance between exploration and exploitation
   
3. **Exploitation Phase** (300 episodes):
   - Low epsilon value
   - Mostly using learned policy

## File Management

### Saved Files
- `models/q_table.json`: Current Q-table
- `models/q_table.json.best`: Best performing model
- `models/training_stats.json`: Training statistics
- `models/training_plot.png`: Learning curve
