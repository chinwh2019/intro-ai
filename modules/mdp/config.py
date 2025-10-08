"""Configuration for MDP Module"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class MDPConfig:
    """Configuration for MDP module"""

    # Display settings
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 800
    CELL_SIZE: int = 100
    FPS: int = 60

    # Grid world settings
    GRID_SIZE: int = 5
    NUM_OBSTACLES: int = 2

    # MDP parameters (students can modify these!)
    DISCOUNT: float = 0.9          # γ (gamma) - discount factor
    NOISE: float = 0.2              # Stochastic transition probability
    LIVING_REWARD: float = -0.04    # Cost of living (per step)

    # Algorithm settings
    VALUE_ITERATION_EPSILON: float = 0.001  # Convergence threshold
    MAX_ITERATIONS: int = 1000
    ANIMATION_SPEED: float = 1.0     # Multiplier for iteration speed
    ITERATION_DELAY: float = 0.5     # Seconds between iterations
    LEARNING_RATE: float = 0.1       # For manual learning mode (α)

    # Visualization settings
    SHOW_VALUES: bool = True
    SHOW_POLICY: bool = True
    SHOW_Q_VALUES: bool = False
    ANIMATE_CONVERGENCE: bool = True
    VALUE_DECIMAL_PLACES: int = 2

    # Colors
    COLOR_BACKGROUND: Tuple[int, int, int] = (30, 30, 46)
    COLOR_GRID_LINE: Tuple[int, int, int] = (68, 71, 90)
    COLOR_OBSTACLE: Tuple[int, int, int] = (88, 91, 112)
    COLOR_GOAL: Tuple[int, int, int] = (255, 215, 0)      # Gold
    COLOR_DANGER: Tuple[int, int, int] = (255, 85, 85)    # Red
    COLOR_AGENT: Tuple[int, int, int] = (0, 150, 255)     # Blue
    COLOR_TEXT: Tuple[int, int, int] = (248, 248, 242)
    COLOR_ARROW: Tuple[int, int, int] = (255, 255, 255)
    COLOR_UI_BG: Tuple[int, int, int] = (40, 42, 54)

    # Value function visualization (heatmap)
    COLOR_VALUE_POSITIVE: Tuple[int, int, int] = (80, 250, 123)  # Green
    COLOR_VALUE_NEGATIVE: Tuple[int, int, int] = (255, 121, 198) # Pink
    COLOR_VALUE_NEUTRAL: Tuple[int, int, int] = (189, 147, 249)  # Purple

    # UI Layout
    SIDEBAR_WIDTH: int = 350
    CONTROL_PANEL_HEIGHT: int = 80


# Global config
config = MDPConfig()


# Presets for different scenarios
PRESETS = {
    'default': MDPConfig(),

    'deterministic': MDPConfig(
        NOISE=0.0,
        DISCOUNT=0.99,
    ),

    'high_noise': MDPConfig(
        NOISE=0.4,
        DISCOUNT=0.9,
    ),

    'cliff_world': MDPConfig(
        GRID_SIZE=8,
        LIVING_REWARD=-1.0,
        DISCOUNT=0.99,
    ),

    'fast_convergence': MDPConfig(
        ANIMATION_SPEED=5.0,
        ITERATION_DELAY=0.1,
    ),
}


def load_preset(name: str):
    """Load a preset configuration"""
    global config
    if name in PRESETS:
        # Get preset config
        preset_config = PRESETS[name]

        # Update all attributes of global config
        for attr in dir(preset_config):
            if not attr.startswith('_') and attr.isupper():
                setattr(config, attr, getattr(preset_config, attr))

        print(f"Loaded preset: {name}")
        print(f"  Grid size: {config.GRID_SIZE}x{config.GRID_SIZE}")
        print(f"  Discount: {config.DISCOUNT}")
        print(f"  Noise: {config.NOISE}")
    else:
        print(f"Unknown preset: {name}")
        print(f"Available presets: {list(PRESETS.keys())}")
