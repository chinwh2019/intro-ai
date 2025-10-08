"""MDP Visualization"""

import pygame
import math
from typing import Dict, Optional
from modules.mdp.config import config
from modules.mdp.environments.grid_world import GridWorld
from modules.mdp.core.mdp import State


class MDPVisualizer:
    """Visualizer for MDP"""

    def __init__(self, grid_world: GridWorld):
        pygame.init()

        self.grid_world = grid_world
        self.grid_size = grid_world.grid_size

        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("MDP Visualization")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 20)
        self.tiny_font = pygame.font.Font(None, 16)

        # Calculate cell size
        available_width = config.WINDOW_WIDTH - config.SIDEBAR_WIDTH
        available_height = config.WINDOW_HEIGHT - config.CONTROL_PANEL_HEIGHT
        self.cell_size = min(
            available_width // self.grid_size,
            available_height // self.grid_size
        )

        # Grid offset for centering
        self.grid_offset_x = config.SIDEBAR_WIDTH + (
            available_width - self.cell_size * self.grid_size
        ) // 2
        self.grid_offset_y = config.CONTROL_PANEL_HEIGHT + (
            available_height - self.cell_size * self.grid_size
        ) // 2

        # Current state
        self.values: Dict[State, float] = {}
        self.q_values: Dict = {}
        self.policy: Dict[State, str] = {}
        self.iteration = 0
        self.converged = False
        self.agent_pos = grid_world.start_pos

    def update_state(self, solver_state: Dict):
        """Update visualization state"""
        self.values = solver_state.get('values', {})
        self.q_values = solver_state.get('q_values', {})
        self.policy = solver_state.get('policy', {})
        self.iteration = solver_state.get('iteration', 0)
        self.converged = solver_state.get('converged', False)

    def render(self):
        """Render MDP"""
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)

        # Render grid
        self._render_grid()

        # Render sidebar
        self._render_sidebar()

        # Render control panel
        self._render_control_panel()

        # Update display
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def _render_grid(self):
        """Render grid world"""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = self.grid_offset_x + c * self.cell_size
                y = self.grid_offset_y + r * self.cell_size

                state = State((r, c))

                # Determine cell type and color
                if (r, c) in self.grid_world.obstacles:
                    color = config.COLOR_OBSTACLE
                    self._draw_cell(x, y, color)

                elif (r, c) == self.grid_world.goal_pos:
                    color = config.COLOR_GOAL
                    self._draw_cell(x, y, color)
                    # Draw star or trophy icon
                    self._draw_goal_icon(x, y)

                elif (r, c) == self.grid_world.danger_pos:
                    color = config.COLOR_DANGER
                    self._draw_cell(x, y, color)
                    # Draw danger icon
                    self._draw_danger_icon(x, y)

                else:
                    # Normal cell - color by value
                    if config.SHOW_VALUES and state in self.values:
                        value = self.values[state]
                        color = self._value_to_color(value)
                        self._draw_cell(x, y, color, alpha=150)
                    else:
                        self._draw_cell(x, y, (50, 50, 50), alpha=50)

                    # Draw value text
                    if config.SHOW_VALUES and state in self.values:
                        value = self.values[state]
                        self._draw_value_text(x, y, value)

                    # Draw policy arrow
                    if config.SHOW_POLICY and state in self.policy:
                        action = self.policy[state]
                        self._draw_policy_arrow(x, y, action)

                    # Draw Q-values in corners
                    if config.SHOW_Q_VALUES:
                        self._draw_q_values(x, y, state)

                # Draw agent if at this position
                if (r, c) == self.agent_pos:
                    self._draw_agent(x, y)

                # Draw grid lines
                pygame.draw.rect(
                    self.screen,
                    config.COLOR_GRID_LINE,
                    (x, y, self.cell_size, self.cell_size),
                    2
                )

    def _draw_cell(self, x: int, y: int, color: tuple, alpha: int = 255):
        """Draw a colored cell"""
        surface = pygame.Surface((self.cell_size, self.cell_size))
        surface.set_alpha(alpha)
        surface.fill(color)
        self.screen.blit(surface, (x, y))

    def _value_to_color(self, value: float) -> tuple:
        """Convert value to color (heatmap)"""
        if value > 0:
            # Positive values: green
            intensity = min(255, int(abs(value) * 255))
            return (0, intensity, 0)
        elif value < 0:
            # Negative values: red
            intensity = min(255, int(abs(value) * 255))
            return (intensity, 0, 0)
        else:
            # Zero: gray
            return (100, 100, 100)

    def _draw_value_text(self, x: int, y: int, value: float):
        """Draw value as text"""
        text = f"{value:.{config.VALUE_DECIMAL_PLACES}f}"
        text_surface = self.small_font.render(text, True, config.COLOR_TEXT)
        text_rect = text_surface.get_rect(
            center=(x + self.cell_size // 2, y + self.cell_size // 2)
        )
        self.screen.blit(text_surface, text_rect)

    def _draw_policy_arrow(self, x: int, y: int, action: str):
        """Draw policy arrow"""
        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2
        arrow_length = self.cell_size // 3

        # Direction vectors
        directions = {
            "UP": (0, -arrow_length),
            "DOWN": (0, arrow_length),
            "LEFT": (-arrow_length, 0),
            "RIGHT": (arrow_length, 0),
        }

        if action in directions:
            dx, dy = directions[action]
            end_x = center_x + dx
            end_y = center_y + dy

            # Draw arrow line
            pygame.draw.line(
                self.screen,
                config.COLOR_ARROW,
                (center_x, center_y),
                (end_x, end_y),
                4
            )

            # Draw arrowhead
            arrow_size = 8
            angle = math.atan2(dy, dx)

            point1_x = end_x - arrow_size * math.cos(angle - math.pi / 6)
            point1_y = end_y - arrow_size * math.sin(angle - math.pi / 6)
            point2_x = end_x - arrow_size * math.cos(angle + math.pi / 6)
            point2_y = end_y - arrow_size * math.sin(angle + math.pi / 6)

            pygame.draw.polygon(
                self.screen,
                config.COLOR_ARROW,
                [(end_x, end_y), (point1_x, point1_y), (point2_x, point2_y)]
            )

    def _draw_q_values(self, x: int, y: int, state: State):
        """Draw Q-values in cell corners"""
        # Q-value positions (top, right, bottom, left)
        positions = {
            "UP": (x + self.cell_size // 2, y + 10),
            "RIGHT": (x + self.cell_size - 25, y + self.cell_size // 2),
            "DOWN": (x + self.cell_size // 2, y + self.cell_size - 10),
            "LEFT": (x + 15, y + self.cell_size // 2),
        }

        for action, pos in positions.items():
            q_value = self.q_values.get((state, action), 0.0)
            text = f"{q_value:.1f}"
            text_surface = self.tiny_font.render(text, True, (200, 200, 200))
            text_rect = text_surface.get_rect(center=pos)
            self.screen.blit(text_surface, text_rect)

    def _draw_goal_icon(self, x: int, y: int):
        """Draw goal icon (star)"""
        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2
        radius = self.cell_size // 4

        # Draw simple circle for now (could be star)
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (center_x, center_y),
            radius,
            3
        )

    def _draw_danger_icon(self, x: int, y: int):
        """Draw danger icon (X)"""
        padding = self.cell_size // 4
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (x + padding, y + padding),
            (x + self.cell_size - padding, y + self.cell_size - padding),
            4
        )
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (x + self.cell_size - padding, y + padding),
            (x + padding, y + self.cell_size - padding),
            4
        )

    def _draw_agent(self, x: int, y: int):
        """Draw agent"""
        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2
        radius = self.cell_size // 4

        pygame.draw.circle(
            self.screen,
            config.COLOR_AGENT,
            (center_x, center_y),
            radius
        )
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (center_x, center_y),
            radius,
            2
        )

    def _render_sidebar(self):
        """Render sidebar with info"""
        # Background
        pygame.draw.rect(
            self.screen,
            config.COLOR_UI_BG,
            (0, 0, config.SIDEBAR_WIDTH, config.WINDOW_HEIGHT)
        )

        # Title
        title = self.font.render("MDP Solver", True, config.COLOR_TEXT)
        self.screen.blit(title, (10, 10))

        # Statistics
        y_offset = 60
        stats = [
            ("Iteration", self.iteration),
            ("Discount (γ)", config.DISCOUNT),
            ("Noise", config.NOISE),
            ("Living Reward", config.LIVING_REWARD),
        ]

        for label, value in stats:
            if isinstance(value, float):
                text = f"{label}: {value:.2f}"
            else:
                text = f"{label}: {value}"
            text_surface = self.small_font.render(text, True, config.COLOR_TEXT)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 30

        # Status
        y_offset += 20
        status_text = "Converged!" if self.converged else "Iterating..."
        status_color = config.COLOR_GOAL if self.converged else config.COLOR_TEXT
        status = self.small_font.render(status_text, True, status_color)
        self.screen.blit(status, (10, y_offset))

        # Legend
        y_offset += 60
        legend_title = self.font.render("Legend:", True, config.COLOR_TEXT)
        self.screen.blit(legend_title, (10, y_offset))
        y_offset += 35

        legend_items = [
            ("Goal (Reward: +1)", config.COLOR_GOAL),
            ("Danger (Reward: -1)", config.COLOR_DANGER),
            ("Obstacle", config.COLOR_OBSTACLE),
            ("Agent", config.COLOR_AGENT),
        ]

        for label, color in legend_items:
            # Color box
            pygame.draw.rect(self.screen, color, (10, y_offset, 20, 20))
            # Label
            text = self.tiny_font.render(label, True, config.COLOR_TEXT)
            self.screen.blit(text, (35, y_offset + 3))
            y_offset += 25

    def _render_control_panel(self):
        """Render control panel"""
        # Background
        pygame.draw.rect(
            self.screen,
            config.COLOR_UI_BG,
            (config.SIDEBAR_WIDTH, 0,
             config.WINDOW_WIDTH - config.SIDEBAR_WIDTH,
             config.CONTROL_PANEL_HEIGHT)
        )

        # Instructions
        instructions = [
            "SPACE: Start/Pause | S: Step | R: Reset | V: Toggle Values | P: Toggle Policy | Q: Toggle Q-values"
        ]

        y_offset = 15
        for instruction in instructions:
            text = self.tiny_font.render(instruction, True, config.COLOR_TEXT)
            self.screen.blit(text, (config.SIDEBAR_WIDTH + 10, y_offset))
            y_offset += 20
