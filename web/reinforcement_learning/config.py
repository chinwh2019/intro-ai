"""
Configuration for Reinforcement Learning Module - Web Version
Students can easily modify these parameters!
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class RLConfig:
    """Configuration for RL module"""

    # Display settings
    WINDOW_WIDTH: int = 1400
    WINDOW_HEIGHT: int = 800
    GAME_WIDTH: int = 800
    GAME_HEIGHT: int = 600
    VIZ_WIDTH: int = 600  # Width of visualization panel
    FPS: int = 60

    # Snake game settings
    BLOCK_SIZE: int = 20
    GAME_SPEED: int = 50  # Steps per second (higher = faster)

    # Q-Learning hyperparameters (students experiment with these!)
    LEARNING_RATE: float = 0.01      # α (alpha) - how fast to learn
    DISCOUNT_FACTOR: float = 0.95    # γ (gamma) - importance of future rewards
    EPSILON_START: float = 1.0       # Initial exploration rate
    EPSILON_END: float = 0.01        # Final exploration rate
    EPSILON_DECAY: float = 0.995     # Decay per episode

    # Training settings
    NUM_EPISODES: int = 1000
    MAX_STEPS_PER_EPISODE: int = 1000
    MEMORY_SIZE: int = 10000         # Experience replay buffer size
    BATCH_SIZE: int = 32             # Mini-batch size for learning

    # Reward shaping (students can modify!)
    REWARD_FOOD: float = 10.0
    REWARD_DEATH: float = -10.0
    REWARD_CLOSER_TO_FOOD: float = 1.0
    REWARD_FARTHER_FROM_FOOD: float = -1.5
    REWARD_IDLE: float = -0.1

    # Visualization settings
    SHOW_Q_VALUES: bool = True
    SHOW_STATE_INFO: bool = True
    SHOW_LEARNING_CURVES: bool = True
    UPDATE_PLOT_EVERY: int = 10      # Episodes between plot updates

    # Colors
    COLOR_BACKGROUND: Tuple[int, int, int] = (30, 30, 46)
    COLOR_SNAKE_HEAD: Tuple[int, int, int] = (0, 200, 100)
    COLOR_SNAKE_BODY: Tuple[int, int, int] = (0, 150, 75)
    COLOR_FOOD: Tuple[int, int, int] = (255, 0, 0)
    COLOR_TEXT: Tuple[int, int, int] = (248, 248, 242)
    COLOR_UI_BG: Tuple[int, int, int] = (40, 42, 54)
    COLOR_GRID: Tuple[int, int, int] = (50, 50, 60)

    # File paths
    MODEL_SAVE_PATH: str = "models/q_table.json"
    STATS_SAVE_PATH: str = "models/stats.json"
    PLOT_SAVE_PATH: str = "models/training_plot.png"


# Global config instance
config = RLConfig()


# Preset configurations
PRESETS = {
    'default': RLConfig(),

    'fast_learning': RLConfig(
        LEARNING_RATE=0.05,
        EPSILON_DECAY=0.99,
        NUM_EPISODES=500,
        GAME_SPEED=150,  # Fast simulation
    ),

    'slow_careful': RLConfig(
        LEARNING_RATE=0.001,
        EPSILON_DECAY=0.999,
        EPSILON_END=0.05,
        NUM_EPISODES=2000,
        GAME_SPEED=20,  # Slow to watch learning
    ),

    'greedy': RLConfig(
        EPSILON_START=0.3,
        EPSILON_END=0.01,
        EPSILON_DECAY=0.99,
    ),

    'turbo': RLConfig(
        LEARNING_RATE=0.02,
        GAME_SPEED=200,  # Maximum speed for quick training
        EPSILON_DECAY=0.98,
        UPDATE_PLOT_EVERY=50,  # Update plots less frequently
    ),
}


def load_preset(name: str):
    """Load a preset configuration"""
    global config
    if name in PRESETS:
        # Get preset config
        preset_config = PRESETS[name]

        # Update all attributes of global config (don't replace the object!)
        # This ensures all references to config stay valid
        for attr in dir(preset_config):
            if not attr.startswith('_') and attr.isupper():
                setattr(config, attr, getattr(preset_config, attr))

        print(f"✓ Loaded preset: {name}")
        print(f"  Learning rate: {config.LEARNING_RATE}")
        print(f"  Game speed: {config.GAME_SPEED}")
    else:
        print(f"✗ Unknown preset: {name}")
        print(f"  Available: {list(PRESETS.keys())}")
