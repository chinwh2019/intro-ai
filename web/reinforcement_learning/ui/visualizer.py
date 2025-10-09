"""RL Visualization - Web Version (No Matplotlib, No NumPy)"""

import pygame
from typing import Optional
from config import config, load_preset
from environments.snake import SnakeEnv
from core.q_learning import QLearningAgent
from utils.stats import TrainingStats
from ui.controls import RLParameterPanel

class RLVisualizer:
    """Visualizer for RL training"""

    def __init__(self, env: SnakeEnv, agent: QLearningAgent):
        print("RLVisualizer init starting...")

        # Initialize pygame (required for web version)
        pygame.init()

        self.env = env
        self.agent = agent
        self.is_inference_mode = False  # Track inference mode for display

        print("Creating display...")
        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Reinforcement Learning - Snake Q-Learning")
        print("✓ Display created")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)

        # Stats for plotting
        self.stats = TrainingStats()
        print("✓ Stats created")

        # Interactive parameter panel
        print("Creating parameter panel...")
        self.parameter_panel = RLParameterPanel(
            x=config.GAME_WIDTH + 10,
            y=10,
            width=config.VIZ_WIDTH - 20,
            on_apply=self.on_parameters_changed
        )
        print("✓ Parameter panel created")

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
            'preset_turbo': ('turbo', 'turbo'),
        }

        if preset_name in preset_map:
            actual_preset, button_name = preset_map[preset_name]

            print(f"\n{'='*60}")
            print(f"📦 Loading preset: {actual_preset}")

            # Load preset (updates config attributes in-place)
            load_preset(actual_preset)

            # Update panel sliders to reflect new values
            print(f"Updating sliders...")
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

            print(f"✓ Preset applied: {actual_preset}")
            print(f"  Learning Rate: {config.LEARNING_RATE}")
            print(f"  Discount: {config.DISCOUNT_FACTOR}")
            print(f"  Epsilon Decay: {config.EPSILON_DECAY}")
            print(f"  Speed: {config.GAME_SPEED}")
            print(f"{'='*60}")
        else:
            print(f"❌ Unknown preset: {preset_name}")

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
        self.is_inference_mode = not is_training
        self.parameter_panel.set_training_mode(is_training)

    def render(self, episode: int = 0, current_state: Optional[tuple] = None, demo_runs: int = 0):
        """Render everything"""
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)

        # Draw game
        self._draw_game()

        # Draw visualization panel (pass demo_runs for display)
        self._draw_viz_panel(episode, current_state, demo_runs)

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

    def _draw_viz_panel(self, episode: int, current_state: Optional[tuple], demo_runs: int = 0):
        """Draw visualization panel"""
        panel_x = config.GAME_WIDTH

        # Draw parameter panel first (at top)
        self.parameter_panel.render(self.screen)

        # Create surface for stats below parameter panel
        stats_y_start = 610  # After parameter panel (which is ~600px tall)
        panel_surface = pygame.Surface((config.VIZ_WIDTH, config.WINDOW_HEIGHT - stats_y_start))
        panel_surface.fill(config.COLOR_UI_BG)

        # TWO-COLUMN LAYOUT for better space usage
        left_col_x = 10
        right_col_x = config.VIZ_WIDTH // 2 + 5  # Start right column at midpoint
        y_offset = 5

        # Title (spans both columns)
        title = self.small_font.render("Current Status", True, config.COLOR_TEXT)
        panel_surface.blit(title, (left_col_x, y_offset))
        y_offset += 22

        # === LEFT COLUMN ===
        left_y = y_offset

        # Episode info (different display for inference vs training)
        if self.is_inference_mode:
            info = f"Demo #{demo_runs}"
        else:
            info = f"Episode: {episode}/{config.NUM_EPISODES}"

        text = self.small_font.render(info, True, config.COLOR_TEXT)
        panel_surface.blit(text, (left_col_x, left_y))
        left_y += 18

        # Game state
        text = self.small_font.render(f"Score: {self.env.score}", True, config.COLOR_TEXT)
        panel_surface.blit(text, (left_col_x, left_y))
        left_y += 18

        text = self.small_font.render(f"Steps: {self.env.frame_count}", True, config.COLOR_TEXT)
        panel_surface.blit(text, (left_col_x, left_y))
        left_y += 22

        # Agent params
        agent_stats = self.agent.get_statistics()
        text = self.small_font.render(f"ε: {agent_stats['epsilon']:.3f}", True, config.COLOR_TEXT)
        panel_surface.blit(text, (left_col_x, left_y))
        left_y += 18

        text = self.small_font.render(f"α: {self.agent.learning_rate:.4f}", True, config.COLOR_TEXT)
        panel_surface.blit(text, (left_col_x, left_y))
        left_y += 18

        text = self.small_font.render(f"Q-size: {agent_stats['q_table_size']}", True, config.COLOR_TEXT)
        panel_surface.blit(text, (left_col_x, left_y))
        left_y += 18

        # === RIGHT COLUMN ===
        right_y = y_offset

        # Q-values for current state
        if config.SHOW_Q_VALUES and current_state is not None:
            q_values = self.agent.get_q_values(current_state)
            action_names = ["Straight", "Right", "Left"]

            text = self.small_font.render("Q-Values:", True, config.COLOR_TEXT)
            panel_surface.blit(text, (right_col_x, right_y))
            right_y += 20

            max_q = max(q_values) if len(q_values) > 0 else 0
            for i, (name, q) in enumerate(zip(action_names, q_values)):
                color = (80, 250, 123) if q == max_q and max_q != 0 else config.COLOR_TEXT
                text = self.small_font.render(
                    f"{name}: {q:.2f}",
                    True,
                    color
                )
                panel_surface.blit(text, (right_col_x, right_y))
                right_y += 18

            right_y += 8  # Separator

        # Training summary - only if not in inference mode
        if not self.is_inference_mode and len(self.stats.episode_scores) > 0:
            summary = self.stats.get_summary()

            text = self.small_font.render("Recent (100 ep):", True, config.COLOR_TEXT)
            panel_surface.blit(text, (right_col_x, right_y))
            right_y += 20

            text = self.small_font.render(f"Avg: {summary['avg_score']:.1f}", True, config.COLOR_TEXT)
            panel_surface.blit(text, (right_col_x, right_y))
            right_y += 18

            text = self.small_font.render(f"Max: {summary['max_score']}", True, config.COLOR_TEXT)
            panel_surface.blit(text, (right_col_x, right_y))
            right_y += 18

            text = self.small_font.render(f"Reward: {summary['avg_reward']:.1f}", True, config.COLOR_TEXT)
            panel_surface.blit(text, (right_col_x, right_y))
            right_y += 18

        # Blit stats panel to screen (below parameter panel)
        self.screen.blit(panel_surface, (panel_x, stats_y_start))

        # Note: Learning curves disabled in web version (matplotlib not compatible with Pygbag)
        # Use desktop version (run_rl.py) for full learning curve visualization
