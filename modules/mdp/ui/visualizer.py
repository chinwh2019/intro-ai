"""MDP Visualization with enhanced features from original implementation"""

import pygame
import math
from typing import Dict, Optional
from modules.mdp.config import config
from modules.mdp.environments.grid_world import GridWorld
from modules.mdp.core.mdp import State


class MDPVisualizer:
    """Enhanced visualizer for MDP with triangular Q-values and better colors"""

    def __init__(self, grid_world: GridWorld):
        pygame.init()

        self.grid_world = grid_world
        self.grid_size = grid_world.grid_size

        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("MDP Visualization")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.tiny_font = pygame.font.Font(None, 16)

        # Calculate cell size to fit in main area
        main_area_size = min(config.WINDOW_WIDTH - config.SIDEBAR_WIDTH,
                            config.WINDOW_HEIGHT - config.CONTROL_PANEL_HEIGHT)
        self.cell_size = main_area_size // self.grid_size

        # Grid offset for centering
        self.grid_offset_x = config.SIDEBAR_WIDTH
        self.grid_offset_y = config.CONTROL_PANEL_HEIGHT

        # Current state
        self.values: Dict[State, float] = {}
        self.q_values: Dict = {}
        self.policy: Dict[State, str] = {}
        self.iteration = 0
        self.converged = False

    def update_state(self, solver_state: Dict):
        """Update visualization state"""
        self.values = solver_state.get('values', {})
        self.q_values = solver_state.get('q_values', {})
        self.policy = solver_state.get('policy', {})
        self.iteration = solver_state.get('iteration', 0)
        self.converged = solver_state.get('converged', False)

    def render(self, walker_pos: Optional[tuple] = None):
        """Render MDP with optional walker position"""
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)

        # Render grid
        self._render_grid(walker_pos)

        # Render sidebar
        self._render_sidebar()

        # Render control panel
        self._render_control_panel()

        # Update display
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def _render_grid(self, walker_pos: Optional[tuple] = None):
        """Render grid world with enhanced Q-value visualization"""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = self.grid_offset_x + c * self.cell_size
                y = self.grid_offset_y + r * self.cell_size

                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                state = State((r, c))

                # Draw based on cell type
                if (r, c) in self.grid_world.obstacles:
                    # Obstacle
                    pygame.draw.rect(self.screen, config.COLOR_OBSTACLE, rect)

                elif (r, c) == self.grid_world.goal_pos:
                    # Goal (treasure)
                    pygame.draw.rect(self.screen, config.COLOR_GOAL, rect)
                    if state in self.values:
                        value_text = self.font.render(f"{self.values[state]:.2f}", True, (0, 0, 0))
                        self.screen.blit(value_text, (rect.centerx - value_text.get_width() // 2,
                                                      rect.centery - value_text.get_height() // 2))

                elif (r, c) == self.grid_world.danger_pos:
                    # Danger (trap)
                    pygame.draw.rect(self.screen, config.COLOR_DANGER, rect)
                    if state in self.values:
                        value_text = self.font.render(f"{self.values[state]:.2f}", True, (255, 255, 255))
                        self.screen.blit(value_text, (rect.centerx - value_text.get_width() // 2,
                                                      rect.centery - value_text.get_height() // 2))

                else:
                    # Normal cell
                    if config.SHOW_Q_VALUES and state in self.policy:
                        # Draw with triangular Q-value partitions
                        self._draw_cell_with_triangles(rect, state)
                    else:
                        # Draw simple cell with value
                        pygame.draw.rect(self.screen, (0, 100, 0), rect)  # DARK_GREEN

                        if config.SHOW_VALUES and state in self.values:
                            value = self.values[state]
                            value_text = self.font.render(f"{value:.2f}", True, (255, 255, 255))
                            self.screen.blit(value_text, (rect.centerx - value_text.get_width() // 2,
                                                          rect.centery - value_text.get_height() // 2))

                    # Draw policy arrow on top
                    if config.SHOW_POLICY and state in self.policy:
                        self._draw_policy_arrow(rect, self.policy[state])

                # Draw grid lines
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

                # Draw walker if at this position
                if walker_pos and (r, c) == walker_pos:
                    pygame.draw.circle(self.screen, (255, 165, 0), rect.center, self.cell_size // 3)

    def _draw_cell_with_triangles(self, rect: pygame.Rect, state: State):
        """Draw cell divided into 4 triangles for Q-values"""
        # Background
        pygame.draw.rect(self.screen, (0, 100, 0), rect)

        # Triangle points for each action (up, right, down, left)
        triangle_points = [
            [(rect.left, rect.top), (rect.right, rect.top), (rect.centerx, rect.centery)],  # UP
            [(rect.right, rect.top), (rect.right, rect.bottom), (rect.centerx, rect.centery)],  # RIGHT
            [(rect.left, rect.bottom), (rect.right, rect.bottom), (rect.centerx, rect.centery)],  # DOWN
            [(rect.left, rect.top), (rect.left, rect.bottom), (rect.centerx, rect.centery)]  # LEFT
        ]

        # Action names matching triangle order
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        # Draw triangles and Q-values
        for i, action in enumerate(action_names):
            q_key = (state, action)
            q_value = self.q_values.get(q_key, 0.0)

            # Color based on Q-value sign
            color = (255, 0, 0) if q_value < 0 else (0, 100, 0)  # RED or DARK_GREEN

            # Draw triangle
            pygame.draw.polygon(self.screen, color, triangle_points[i])
            pygame.draw.lines(self.screen, (255, 255, 255), True, triangle_points[i], 1)

            # Draw Q-value text
            q_text = self.font.render(f"{q_value:.2f}", True, (255, 255, 255))

            # Position text in triangle
            if action == 'UP':
                text_pos = (rect.centerx - q_text.get_width() // 2, rect.top + 5)
            elif action == 'RIGHT':
                text_pos = (rect.right - q_text.get_width() - 5, rect.centery - q_text.get_height() // 2)
            elif action == 'DOWN':
                text_pos = (rect.centerx - q_text.get_width() // 2, rect.bottom - q_text.get_height() - 5)
            else:  # LEFT
                text_pos = (rect.left + 5, rect.centery - q_text.get_height() // 2)

            self.screen.blit(q_text, text_pos)

    def _draw_policy_arrow(self, rect: pygame.Rect, action: str):
        """Draw policy arrow"""
        center_x = rect.centerx
        center_y = rect.centery
        arrow_length = self.cell_size // 3

        # Arrow points based on action
        arrow_points = {
            'UP': [(center_x, rect.top + 5), (center_x - 5, rect.top + 15), (center_x + 5, rect.top + 15)],
            'DOWN': [(center_x, rect.bottom - 5), (center_x - 5, rect.bottom - 15), (center_x + 5, rect.bottom - 15)],
            'LEFT': [(rect.left + 5, center_y), (rect.left + 15, center_y - 5), (rect.left + 15, center_y + 5)],
            'RIGHT': [(rect.right - 5, center_y), (rect.right - 15, center_y - 5), (rect.right - 15, center_y + 5)]
        }

        if action in arrow_points:
            pygame.draw.polygon(self.screen, (255, 255, 255), arrow_points[action])

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
            ("Walker", (255, 165, 0)),
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

        # Instructions (split into two lines for clarity)
        instructions_line1 = self.tiny_font.render(
            "L: Learning Animation | SPACE: Pause | S: Step | D: Policy Demo | R: Reset",
            True,
            config.COLOR_TEXT
        )
        instructions_line2 = self.tiny_font.render(
            "V: Values | P: Policy | Q: Q-Values (Triangular) | Arrow Keys: Manual Control",
            True,
            config.COLOR_TEXT
        )

        self.screen.blit(
            instructions_line1,
            (config.SIDEBAR_WIDTH + 10, config.CONTROL_PANEL_HEIGHT // 2 - 20)
        )
        self.screen.blit(
            instructions_line2,
            (config.SIDEBAR_WIDTH + 10, config.CONTROL_PANEL_HEIGHT // 2 + 5)
        )

    def draw_buttons(self, buttons: dict, button_states: dict):
        """Draw interactive buttons at bottom"""
        for button_name, button_rect in buttons.items():
            # Button color (highlight if active)
            color = (150, 200, 150) if button_states.get(button_name, False) else (200, 200, 200)
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), button_rect, 2)

            # Button text
            button_texts = {
                'policy': "Toggle Policy",
                'q_values': "Toggle Q-Values",
                'learning': "Show Learning",
                'manual': "Manual Learning",
                'demo': "Policy Demo"
            }

            text = self.font.render(button_texts.get(button_name, button_name), True, (0, 0, 0))
            self.screen.blit(text, (button_rect.x + 10, button_rect.y + 5))

    def draw_status_message(self, message: str, y_pos: int):
        """Draw status message"""
        if message:
            text = self.font.render(message, True, (255, 0, 0))
            self.screen.blit(text, (10, y_pos))
