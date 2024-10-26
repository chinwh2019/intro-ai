from dataclasses import dataclass

@dataclass
class GameConfig:
    WIDTH: int = 800
    HEIGHT: int = 600
    GRAVITY: float = 0.5
    FLAP_STRENGTH: float = -8
    PIPE_SPEED: float = 3
    PIPE_SPACING: float = 300
    FPS: int = 60

@dataclass
class Colors:
    WHITE: tuple = (255, 255, 255)
    BLACK: tuple = (0, 0, 0)
    GREEN: tuple = (0, 255, 0)
    BLUE: tuple = (0, 0, 255)
    RED: tuple = (255, 0, 0)

@dataclass
class ExperimentConfig:
    learning_rate: float
    discount_factor: float
    epsilon: float
    epsilon_decay: float
    min_epsilon: float