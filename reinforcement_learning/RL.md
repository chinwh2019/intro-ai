# Reinforcement Learning 

This code demonstrates Q-learning in a simple game environment.

## Setup and Installation

1. Directory structure:
   ```
   reinforcement_learning/
   ├── config.py
   ├── game_environment.py
   ├── q_learning_agent.py
   ├── experiments.py
   ├── main.py
   └── data/
   ```

2. Create a `data` directory for Q-table storage.

3. Install required packages:
   ```bash
   pip install -r requirements.txt

4. Run the program:
   ```bash
   python main.py
   ```

## Controls

- `SPACE`: Switch between AI and Player mode
- `UP`: Flap in Player mode
- `E`: Run learning rate experiment

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `config.py` | Configuration settings and constants |
| `game_environment.py` | Game mechanics and rendering |
| `q_learning_agent.py` | Q-learning algorithm implementation |
| `experiments.py` | Experimental features and analysis |
| `main.py` | Program entry point and main loop |

## Getting Started

1. Ensure you have Python 3.x installed.
2. Follow the setup instructions above.
3. Run `main.py` to start the program.
4. Use the controls to interact with the game and AI.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.