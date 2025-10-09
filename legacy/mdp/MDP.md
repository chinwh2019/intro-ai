# Treasure and Trap MDP Visualization

## Overview

This project visualizes a Markov Decision Process (MDP) in a grid world environment with a treasure and a trap. The agent can move in four directions and aims to reach the treasure while avoiding the trap. The visualization includes features to display value functions, policies, and Q-values.

## File Structure

- `main.py`: Entry point of the application.
- `mdp.py`: Contains the MDP class and functions related to MDP creation and value iteration.
- `agent.py`: Defines the Agent class, handling the agent's actions and state.
- `visualizer.py`: Contains the GridWorldVisualizer class for rendering the grid world.
- `utils.py`: Includes utility functions and constants used across modules.
- `README.md`: Project documentation.

## Features

- **Value Iteration Visualization**: Shows the convergence of state values through value iteration.
- **Policy Display**: Toggle to display the optimal policy derived from value iteration.
- **Q-Values Display**: Toggle to display Q-values for each action in each state.
- **Manual Learning Mode**: Manually control the agent to see how Q-values update.
- **Policy Demo Mode**: Watch the agent follow the optimal policy automatically.
- **Notifications**: Receive notifications when the agent reaches the treasure or falls into the trap.

## Controls

- **Mouse Clicks**:
  - Click on buttons to toggle features and modes.
- **Keyboard Shortcuts**:
  - `SPACE` or `P`: Toggle policy display.
  - `Q`: Toggle Q-values display.
  - `L`: Show learning process of value iteration.
  - `R`: Randomize the maze (generate new positions for treasure, trap, and obstacles).
  - Arrow Keys: Control the agent in manual learning mode.

## How to Run

1. Ensure you have Python 3.x installed.
2. Install the required packages:
   ```bash
   pip install pygame
