"""RL Visualization"""

import pygame
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from typing import Optional
from modules.reinforcement_learning.config import config
from modules.reinforcement_learning.environments.snake import SnakeEnv
from modules.reinforcement_learning.core.q_learning import QLearningAgent
from modules.reinforcement_learning.utils.stats import TrainingStats

class RLVisualizer:
    """Visualizer for RL training"""

    def __init__(self, env: SnakeEnv, agent: QLearningAgent):
        pygame.init()

        self.env = env
        self.agent = agent

        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Reinforcement Learning - Snake Q-Learning")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)

        # Stats for plotting
        self.stats = TrainingStats()

        # Matplotlib figure for learning curves
        self.fig, self.axes = plt.subplots(2, 1, figsize=(6, 5))
        self.canvas = FigureCanvasAgg(self.fig)

    def render(self, episode: int = 0, current_state: Optional[np.ndarray] = None):
        """Render everything"""
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)

        # Draw game
        self._draw_game()

        # Draw visualization panel
        self._draw_viz_panel(episode, current_state)

        # Update display
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def _draw_game(self):
        """Draw snake game"""
        game_surface = pygame.Surface((config.GAME_WIDTH, config.GAME_HEIGHT))
        game_surface.fill(config.COLOR_BACKGROUND)

        # Draw grid
        for x in range(0, config.GAME_WIDTH, config.BLOCK_SIZE):
            pygame.draw.line(
                game_surface,
                config.COLOR_GRID,
                (x, 0),
                (x, config.GAME_HEIGHT)
            )
        for y in range(0, config.GAME_HEIGHT, config.BLOCK_SIZE):
            pygame.draw.line(
                game_surface,
                config.COLOR_GRID,
                (0, y),
                (config.GAME_WIDTH, y)
            )

        # Draw food
        pygame.draw.rect(
            game_surface,
            config.COLOR_FOOD,
            pygame.Rect(
                self.env.food[0],
                self.env.food[1],
                config.BLOCK_SIZE,
                config.BLOCK_SIZE
            )
        )

        # Draw snake
        for i, segment in enumerate(self.env.snake):
            color = config.COLOR_SNAKE_HEAD if i == 0 else config.COLOR_SNAKE_BODY
            pygame.draw.rect(
                game_surface,
                color,
                pygame.Rect(
                    segment[0],
                    segment[1],
                    config.BLOCK_SIZE,
                    config.BLOCK_SIZE
                )
            )
            # Border
            pygame.draw.rect(
                game_surface,
                (255, 255, 255),
                pygame.Rect(
                    segment[0],
                    segment[1],
                    config.BLOCK_SIZE,
                    config.BLOCK_SIZE
                ),
                1
            )

        self.screen.blit(game_surface, (0, 0))

    def _draw_viz_panel(self, episode: int, current_state: Optional[np.ndarray]):
        """Draw visualization panel"""
        panel_x = config.GAME_WIDTH
        panel_surface = pygame.Surface((config.VIZ_WIDTH, config.WINDOW_HEIGHT))
        panel_surface.fill(config.COLOR_UI_BG)

        y_offset = 10

        # Title
        title = self.font.render("Q-Learning", True, config.COLOR_TEXT)
        panel_surface.blit(title, (10, y_offset))
        y_offset += 40

        # Episode info
        info_lines = [
            f"Episode: {episode}/{config.NUM_EPISODES}",
            f"Score: {self.env.score}",
            f"Steps: {self.env.frame_count}",
        ]

        for line in info_lines:
            text = self.small_font.render(line, True, config.COLOR_TEXT)
            panel_surface.blit(text, (10, y_offset))
            y_offset += 25

        y_offset += 10

        # Agent stats
        agent_stats = self.agent.get_statistics()
        stats_lines = [
            f"ε (exploration): {agent_stats['epsilon']:.3f}",
            f"Learning rate: {self.agent.learning_rate:.3f}",
            f"Q-table size: {agent_stats['q_table_size']}",
            f"Explore ratio: {agent_stats['exploration_ratio']:.2f}",
        ]

        for line in stats_lines:
            text = self.small_font.render(line, True, config.COLOR_TEXT)
            panel_surface.blit(text, (10, y_offset))
            y_offset += 25

        y_offset += 10

        # Q-values for current state
        if config.SHOW_Q_VALUES and current_state is not None:
            q_values = self.agent.get_q_values(current_state)
            action_names = ["Straight", "Right", "Left"]

            text = self.font.render("Q-Values:", True, config.COLOR_TEXT)
            panel_surface.blit(text, (10, y_offset))
            y_offset += 30

            max_q = max(q_values)
            for i, (name, q) in enumerate(zip(action_names, q_values)):
                color = (0, 255, 0) if q == max_q else config.COLOR_TEXT
                text = self.small_font.render(
                    f"{name}: {q:.2f}",
                    True,
                    color
                )
                panel_surface.blit(text, (10, y_offset))
                y_offset += 23

            y_offset += 10

        # Training summary
        if len(self.stats.episode_scores) > 0:
            summary = self.stats.get_summary()

            text = self.font.render("Recent Performance:", True, config.COLOR_TEXT)
            panel_surface.blit(text, (10, y_offset))
            y_offset += 30

            summary_lines = [
                f"Avg Score: {summary['avg_score']:.1f}",
                f"Max Score: {summary['max_score']}",
                f"Avg Reward: {summary['avg_reward']:.1f}",
            ]

            for line in summary_lines:
                text = self.small_font.render(line, True, config.COLOR_TEXT)
                panel_surface.blit(text, (10, y_offset))
                y_offset += 23

        # Blit panel to screen
        self.screen.blit(panel_surface, (panel_x, 0))

        # Draw learning curves (if enough data)
        if len(self.stats.episode_scores) > 10:
            self._draw_learning_curves(episode)

    def _draw_learning_curves(self, episode: int):
        """Draw learning curves using matplotlib"""
        if episode % config.UPDATE_PLOT_EVERY != 0:
            return  # Don't update every frame

        # Clear axes
        for ax in self.axes:
            ax.clear()

        # Plot scores
        self.axes[0].plot(self.stats.episode_scores, alpha=0.3, color='blue')
        if self.stats.avg_scores:
            self.axes[0].plot(range(len(self.stats.avg_scores), len(self.stats.avg_scores) + len(self.stats.avg_scores)),
                            self.stats.avg_scores, color='red', linewidth=2, label='Avg')
        self.axes[0].set_title('Score per Episode')
        self.axes[0].set_xlabel('Episode')
        self.axes[0].set_ylabel('Score')
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)

        # Plot rewards
        self.axes[1].plot(self.stats.episode_rewards, alpha=0.3, color='green')
        if self.stats.avg_rewards:
            self.axes[1].plot(range(len(self.stats.avg_rewards), len(self.stats.avg_rewards) + len(self.stats.avg_rewards)),
                            self.stats.avg_rewards, color='red', linewidth=2, label='Avg')
        self.axes[1].set_title('Total Reward per Episode')
        self.axes[1].set_xlabel('Episode')
        self.axes[1].set_ylabel('Reward')
        self.axes[1].legend()
        self.axes[1].grid(True, alpha=0.3)

        self.fig.tight_layout()

        # Convert to pygame surface
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()

        plot_surface = pygame.image.fromstring(raw_data, size, "RGB")

        # Blit to screen
        plot_x = config.GAME_WIDTH
        plot_y = config.WINDOW_HEIGHT - plot_surface.get_height()
        self.screen.blit(plot_surface, (plot_x, plot_y))
