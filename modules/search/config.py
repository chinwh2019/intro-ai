"""
Configuration for Search Algorithms Module
Students can modify parameters here to change behavior
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class SearchConfig:
    """Configuration for search module"""

    # Display settings
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 700
    CELL_SIZE: int = 20
    FPS: int = 60

    # Maze settings
    MAZE_WIDTH: int = 40
    MAZE_HEIGHT: int = 30
    MAZE_COMPLEXITY: float = 0.75  # 0.0 to 1.0 (higher = more walls)

    # Start and end point settings (None = auto-generate)
    START_POSITION: Optional[Tuple[int, int]] = None  # (row, col) or None for auto
    GOAL_POSITION: Optional[Tuple[int, int]] = None   # (row, col) or None for auto
    RANDOM_START_GOAL: bool = False  # Randomize start/goal on each reset

    # Algorithm settings
    ANIMATION_SPEED: float = 1.0  # Multiplier for step delay
    STEP_DELAY: float = 0.01      # Seconds between steps (at speed 1.0)
    SHOW_FRONTIER: bool = True
    SHOW_EXPLORED: bool = True
    SHOW_NUMBERS: bool = False    # Show f/g/h values

    # Colors (RGB tuples)
    COLOR_BACKGROUND: Tuple[int, int, int] = (30, 30, 46)
    COLOR_WALL: Tuple[int, int, int] = (69, 71, 90)
    COLOR_PATH: Tuple[int, int, int] = (0, 255, 255)
    COLOR_START: Tuple[int, int, int] = (0, 255, 0)
    COLOR_GOAL: Tuple[int, int, int] = (255, 0, 0)
    COLOR_EXPLORED: Tuple[int, int, int] = (255, 255, 100)
    COLOR_FRONTIER: Tuple[int, int, int] = (167, 139, 250)
    COLOR_CURRENT: Tuple[int, int, int] = (255, 165, 0)
    COLOR_TEXT: Tuple[int, int, int] = (248, 248, 242)
    COLOR_UI_BG: Tuple[int, int, int] = (40, 42, 54)
    COLOR_UI_BORDER: Tuple[int, int, int] = (68, 71, 90)
    COLOR_BUTTON: Tuple[int, int, int] = (98, 114, 164)
    COLOR_BUTTON_HOVER: Tuple[int, int, int] = (139, 157, 216)
    COLOR_BUTTON_ACTIVE: Tuple[int, int, int] = (80, 250, 123)

    # UI Layout
    SIDEBAR_WIDTH: int = 300
    CONTROL_PANEL_HEIGHT: int = 60

    def get_maze_rect(self) -> Tuple[int, int, int, int]:
        """Get rectangle for maze area (x, y, width, height)"""
        return (
            self.SIDEBAR_WIDTH,
            self.CONTROL_PANEL_HEIGHT,
            self.WINDOW_WIDTH - self.SIDEBAR_WIDTH,
            self.WINDOW_HEIGHT - self.CONTROL_PANEL_HEIGHT
        )


# Global config instance (students can modify)
config = SearchConfig()


# Preset configurations for different scenarios
PRESETS = {
    'default': SearchConfig(),

    'fast': SearchConfig(
        ANIMATION_SPEED=5.0,
        STEP_DELAY=0.001,
    ),

    'detailed': SearchConfig(
        ANIMATION_SPEED=0.3,
        STEP_DELAY=0.05,
        SHOW_NUMBERS=True,
        CELL_SIZE=30,
        MAZE_WIDTH=30,
        MAZE_HEIGHT=20,
    ),

    'large_maze': SearchConfig(
        MAZE_WIDTH=60,
        MAZE_HEIGHT=40,
        CELL_SIZE=15,
        WINDOW_WIDTH=1400,
        WINDOW_HEIGHT=800,
    ),

    'simple': SearchConfig(
        MAZE_WIDTH=20,
        MAZE_HEIGHT=15,
        MAZE_COMPLEXITY=0.3,
        CELL_SIZE=30,
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
        print(f"  Maze size: {config.MAZE_WIDTH}x{config.MAZE_HEIGHT}")
    else:
        print(f"Unknown preset: {name}")
        print(f"Available presets: {list(PRESETS.keys())}")
