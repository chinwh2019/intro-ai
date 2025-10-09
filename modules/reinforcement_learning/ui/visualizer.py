"""RL Visualization"""

import pygame
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from typing import Optional
from modules.reinforcement_learning.config import config, load_preset
from modules.reinforcement_learning.environments.snake import SnakeEnv
from modules.reinforcement_learning.core.q_learning import QLearningAgent
from modules.reinforcement_learning.utils.stats import TrainingStats
from modules.reinforcement_learning.ui.controls import RLParameterPanel

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

        # Interactive parameter panel
        self.parameter_panel = RLParameterPanel(
            x=config.GAME_WIDTH + 10,
            y=10,
            width=config.VIZ_WIDTH - 20,
            on_apply=self.on_parameters_changed
        )

        # Set initial parameter values
        self.parameter_panel.set_parameters(
            learning_rate=config.LEARNING_RATE,
            discount=config.DISCOUNT_FACTOR,
            epsilon_start=config.EPSILON_START,
            epsilon_decay=config.EPSILON_DECAY,
            speed=config.GAME_SPEED
        )

    def on_parameters_changed(self, params: dict):
        """Handle parameter changes from UI"""
        print("\n" + "=" * 60)
        print("🔄 Applying new training parameters...")
        print(f"  Learning Rate: {params['learning_rate']:.4f}")
        print(f"  Discount: {params['discount']:.3f}")
        print(f"  Epsilon Start: {params['epsilon_start']:.2f}")
        print(f"  Epsilon Decay: {params['epsilon_decay']:.4f}")
        print(f"  Speed: {params['speed']:.0f}")

        # Update config
        config.LEARNING_RATE = params['learning_rate']
        config.DISCOUNT_FACTOR = params['discount']
        config.EPSILON_START = params['epsilon_start']
        config.EPSILON_DECAY = params['epsilon_decay']
        config.GAME_SPEED = params['speed']

        # Update agent parameters
        self.agent.learning_rate = params['learning_rate']
        self.agent.discount = params['discount']
        # Note: epsilon_start and decay affect future training, not current epsilon
        self.agent.epsilon_decay = params['epsilon_decay']

        print("✓ Parameters updated!")
        print("=" * 60)

    def on_preset_selected(self, preset_name: str):
        """Handle preset selection"""
        preset_map = {
            'preset_default': ('default', 'default'),
            'preset_fast': ('fast_learning', 'fast'),
            'preset_slow': ('slow_careful', 'slow'),
            'preset_demo': ('visual_demo', 'demo'),
        }

        if preset_name in preset_map:
            actual_preset, button_name = preset_map[preset_name]

            print(f"\n{'='*60}")
            print(f"📦 Loading preset: {actual_preset}")
            load_preset(actual_preset)

            # Update panel to reflect new values
            self.parameter_panel.set_parameters(
                learning_rate=config.LEARNING_RATE,
                discount=config.DISCOUNT_FACTOR,
                epsilon_start=config.EPSILON_START,
                epsilon_decay=config.EPSILON_DECAY,
                speed=config.GAME_SPEED
            )

            # Set active preset indicator
            self.parameter_panel.set_active_preset(button_name)

            # Apply to agent immediately
            self.agent.learning_rate = config.LEARNING_RATE
            self.agent.discount = config.DISCOUNT_FACTOR
            self.agent.epsilon_decay = config.EPSILON_DECAY

            print(f"✓ Preset loaded: {actual_preset}")
            print(f"  Learning Rate: {config.LEARNING_RATE}")
            print(f"  Discount: {config.DISCOUNT_FACTOR}")
            print(f"  Epsilon Decay: {config.EPSILON_DECAY}")
            print(f"  Speed: {config.GAME_SPEED}")
            print(f"{'='*60}")

    def handle_parameter_event(self, event: pygame.event.Event):
        """Handle events for parameter panel"""
        action = self.parameter_panel.handle_event(event)
        if action:
            if action.startswith('preset_'):
                self.on_preset_selected(action)
            elif action == 'apply':
                params = self.parameter_panel.get_parameters()
                self.on_parameters_changed(params)
            elif action == 'save':
                return 'save_model'
            elif action == 'load':
                return 'load_model'
            elif action == 'toggle_training':
                return 'toggle_training'
        return None

    def set_training_mode(self, is_training: bool):
        """Update UI to reflect training mode"""
        self.parameter_panel.set_training_mode(is_training)

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

        # Draw parameter panel first (at top)
        self.parameter_panel.render(self.screen)

        # Create surface for stats below parameter panel
        stats_y_start = 610  # After parameter panel (which is ~600px tall now)
        panel_surface = pygame.Surface((config.VIZ_WIDTH, config.WINDOW_HEIGHT - stats_y_start))
        panel_surface.fill(config.COLOR_UI_BG)

        y_offset = 10

        # Title
        title = self.font.render("Stats & Q-Values", True, config.COLOR_TEXT)
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

        # Blit stats panel to screen (below parameter panel)
        self.screen.blit(panel_surface, (panel_x, stats_y_start))

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
        self.axes[0].plot(self.stats.episode_scores, alpha=0.3, color='blue', label='Score')
        if self.stats.avg_scores:
            # Moving average starts at window size (100 episodes)
            window = 100
            x_avg = range(window, window + len(self.stats.avg_scores))
            self.axes[0].plot(x_avg, self.stats.avg_scores, color='red', linewidth=2, label='Avg (100 ep)')
        self.axes[0].set_title('Score per Episode')
        self.axes[0].set_xlabel('Episode')
        self.axes[0].set_ylabel('Score')
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)

        # Plot rewards
        self.axes[1].plot(self.stats.episode_rewards, alpha=0.3, color='green', label='Reward')
        if self.stats.avg_rewards:
            # Moving average starts at window size (100 episodes)
            window = 100
            x_avg = range(window, window + len(self.stats.avg_rewards))
            self.axes[1].plot(x_avg, self.stats.avg_rewards, color='red', linewidth=2, label='Avg (100 ep)')
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
