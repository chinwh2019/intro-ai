"""
Visualization system for search algorithms
"""

import pygame
from typing import Optional, Dict, Set, Callable
from config import config
from core.environment import Maze
from core.state import State
from ui.controls import SearchParameterPanel

class SearchVisualizer:
    """Visualizer for search algorithms"""

    def __init__(self, maze: Maze, on_parameter_change: Callable = None):
        pygame.init()

        self.maze = maze
        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Search Algorithms Visualization")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        self.tiny_font = pygame.font.Font(None, 14)  # For compact legend

        # Calculate cell size based on maze dimensions
        maze_rect = config.get_maze_rect()
        self.cell_width = maze_rect[2] // maze.width
        self.cell_height = maze_rect[3] // maze.height
        self.cell_size = min(self.cell_width, self.cell_height)

        # Offset for centering maze
        self.maze_offset_x = maze_rect[0]
        self.maze_offset_y = maze_rect[1]

        # Current visualization state
        self.explored: Set[State] = set()
        self.frontier: Set[State] = set()
        self.current_state: Optional[State] = None
        self.path: list = []
        self.solution_found = False
        self.stats: Dict = {}
        self.current_algorithm_name: str = "None"  # Track current algorithm

        # Interactive parameter panel (positioned to avoid overlap with legend)
        self.parameter_panel = SearchParameterPanel(
            x=10,
            y=config.WINDOW_HEIGHT - 330,  # Moved down to avoid legend overlap
            width=config.SIDEBAR_WIDTH - 20,
            on_apply=on_parameter_change
        )

        # Set initial values from config
        self.parameter_panel.set_parameters(
            speed=config.ANIMATION_SPEED,
            heuristic_weight=1.0,  # Default
            complexity=config.MAZE_COMPLEXITY
        )

    def update_state(self, viz_state: Dict):
        """Update visualization state"""
        self.explored = viz_state.get('explored', set())
        self.frontier = viz_state.get('frontier', set())
        self.current_state = viz_state.get('current')
        self.path = viz_state.get('path', [])
        self.solution_found = viz_state.get('solution_found', False)
        self.stats = viz_state.get('stats', {})

    def set_algorithm(self, algorithm_name: str):
        """Set the current algorithm name for display"""
        self.current_algorithm_name = algorithm_name

    def render(self):
        """Render current state"""
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)

        # Render maze
        self._render_maze()

        # Render explored, frontier, and path
        self._render_search_state()

        # Render UI
        self._render_sidebar()
        self._render_control_panel()

        # Update display
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def _render_maze(self):
        """Render maze grid"""
        for row in range(self.maze.height):
            for col in range(self.maze.width):
                x = self.maze_offset_x + col * self.cell_size
                y = self.maze_offset_y + row * self.cell_size

                # Determine cell color
                if self.maze.grid[row][col] == 1:
                    color = config.COLOR_WALL
                else:
                    color = config.COLOR_BACKGROUND

                # Draw cell
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x, y, self.cell_size, self.cell_size)
                )

                # Draw grid lines
                pygame.draw.rect(
                    self.screen,
                    (50, 50, 50),
                    (x, y, self.cell_size, self.cell_size),
                    1
                )

    def _render_search_state(self):
        """Render search algorithm state"""
        # Render explored states
        if config.SHOW_EXPLORED:
            for state in self.explored:
                self._draw_cell(state.position, config.COLOR_EXPLORED, alpha=100)

        # Render frontier
        if config.SHOW_FRONTIER:
            for state in self.frontier:
                self._draw_cell(state.position, config.COLOR_FRONTIER, alpha=150)

        # Render path
        for state in self.path:
            self._draw_cell(state.position, config.COLOR_PATH, alpha=200)

        # Render current state
        if self.current_state:
            self._draw_cell(
                self.current_state.position,
                config.COLOR_CURRENT,
                alpha=255
            )

        # Render start and goal
        self._draw_cell(self.maze.start, config.COLOR_START, alpha=255)
        self._draw_cell(self.maze.goal, config.COLOR_GOAL, alpha=255)

    def _draw_cell(self, position: tuple, color: tuple, alpha: int = 255):
        """Draw a colored cell"""
        row, col = position
        x = self.maze_offset_x + col * self.cell_size
        y = self.maze_offset_y + row * self.cell_size

        # Create surface with alpha
        surface = pygame.Surface((self.cell_size, self.cell_size))
        surface.set_alpha(alpha)
        surface.fill(color)

        self.screen.blit(surface, (x, y))

    def _render_sidebar(self):
        """Render sidebar with statistics"""
        # Background
        pygame.draw.rect(
            self.screen,
            config.COLOR_UI_BG,
            (0, 0, config.SIDEBAR_WIDTH, config.WINDOW_HEIGHT)
        )

        # Border
        pygame.draw.line(
            self.screen,
            config.COLOR_UI_BORDER,
            (config.SIDEBAR_WIDTH, 0),
            (config.SIDEBAR_WIDTH, config.WINDOW_HEIGHT),
            2
        )

        # Title
        title = self.font.render("Search Statistics", True, config.COLOR_TEXT)
        self.screen.blit(title, (10, 10))

        # Current Algorithm indicator
        y_offset = 40
        algo_label = self.small_font.render("Algorithm:", True, config.COLOR_TEXT)
        self.screen.blit(algo_label, (10, y_offset))

        algo_name_color = config.COLOR_BUTTON_ACTIVE if self.current_algorithm_name != "None" else config.COLOR_TEXT
        algo_name = self.font.render(self.current_algorithm_name, True, algo_name_color)
        self.screen.blit(algo_name, (10, y_offset + 18))

        # Algorithm selection guide - show key mappings
        y_offset = 75
        keys_title = self.tiny_font.render("Select Algorithm:", True, (180, 180, 180))
        self.screen.blit(keys_title, (10, y_offset))
        y_offset += 16

        key_mappings = [
            "1: BFS      2: DFS",
            "3: UCS      4: A*",
            "5: Greedy"
        ]
        for mapping in key_mappings:
            keys_text = self.tiny_font.render(mapping, True, (150, 150, 150))
            self.screen.blit(keys_text, (10, y_offset))
            y_offset += 14

        # Legend - Moved to TOP after title (compact 2-column)
        y_offset += 10
        legend_title = self.small_font.render("Legend:", True, config.COLOR_TEXT)
        self.screen.blit(legend_title, (10, y_offset))
        y_offset += 20

        # Two columns to save space
        legend_items_col1 = [
            ("Start", config.COLOR_START),
            ("Goal", config.COLOR_GOAL),
            ("Explored", config.COLOR_EXPLORED),
        ]
        legend_items_col2 = [
            ("Frontier", config.COLOR_FRONTIER),
            ("Path", config.COLOR_PATH),
            ("Current", config.COLOR_CURRENT),
        ]

        # Draw column 1
        legend_y = y_offset
        for label, color in legend_items_col1:
            pygame.draw.rect(self.screen, color, (10, legend_y, 12, 12))
            text = self.tiny_font.render(label, True, config.COLOR_TEXT)
            self.screen.blit(text, (25, legend_y + 1))
            legend_y += 16

        # Draw column 2
        legend_y = y_offset
        for label, color in legend_items_col2:
            pygame.draw.rect(self.screen, color, (140, legend_y, 12, 12))
            text = self.tiny_font.render(label, True, config.COLOR_TEXT)
            self.screen.blit(text, (155, legend_y + 1))
            legend_y += 16

        # Statistics - AFTER legend
        y_offset = 215  # Fixed position after legend
        stats_to_show = [
            ("Nodes Expanded", self.stats.get('nodes_expanded', 0)),
            ("Nodes Generated", self.stats.get('nodes_generated', 0)),
            ("Max Frontier", self.stats.get('max_frontier_size', 0)),
            ("Steps", self.stats.get('steps', 0)),
            ("Path Length", self.stats.get('solution_length', 0)),
        ]

        for label, value in stats_to_show:
            text = self.small_font.render(
                f"{label}: {value}",
                True,
                config.COLOR_TEXT
            )
            self.screen.blit(text, (10, y_offset))
            y_offset += 25

        # Status
        y_offset += 20
        if self.solution_found:
            status_text = "Solution Found!"
            status_color = config.COLOR_START
        else:
            status_text = "Searching..."
            status_color = config.COLOR_TEXT

        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (10, y_offset))

        # Random mode indicator
        y_offset += 40
        random_mode = "Random: ON" if config.RANDOM_START_GOAL else "Random: OFF"
        random_color = config.COLOR_BUTTON_ACTIVE if config.RANDOM_START_GOAL else config.COLOR_TEXT
        random_text = self.small_font.render(random_mode, True, random_color)
        self.screen.blit(random_text, (10, y_offset))

        # Draw interactive parameter panel at bottom (legend already at top)
        self.parameter_panel.draw(self.screen)

    def handle_parameter_event(self, event: pygame.event.Event):
        """Pass event to parameter panel"""
        self.parameter_panel.handle_event(event)

    def _render_control_panel(self):
        """Render top control panel"""
        # Background
        pygame.draw.rect(
            self.screen,
            config.COLOR_UI_BG,
            (config.SIDEBAR_WIDTH, 0,
             config.WINDOW_WIDTH - config.SIDEBAR_WIDTH,
             config.CONTROL_PANEL_HEIGHT)
        )

        # Border
        pygame.draw.line(
            self.screen,
            config.COLOR_UI_BORDER,
            (config.SIDEBAR_WIDTH, config.CONTROL_PANEL_HEIGHT),
            (config.WINDOW_WIDTH, config.CONTROL_PANEL_HEIGHT),
            2
        )

        # Instructions (split into two lines)
        instructions_line1 = self.small_font.render(
            "1-5: Select Algorithm | SPACE: Pause/Resume | S: Step | R: Reset",
            True,
            config.COLOR_TEXT
        )
        instructions_line2 = self.small_font.render(
            "T: Toggle Random Start/Goal | Q: Quit",
            True,
            config.COLOR_TEXT
        )

        self.screen.blit(
            instructions_line1,
            (config.SIDEBAR_WIDTH + 10, config.CONTROL_PANEL_HEIGHT // 2 - 18)
        )
        self.screen.blit(
            instructions_line2,
            (config.SIDEBAR_WIDTH + 10, config.CONTROL_PANEL_HEIGHT // 2 + 5)
        )
