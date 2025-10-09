"""
Reinforcement Learning Module - Web Version
Runs in browser via Pygbag/WebAssembly

Controls:
  SPACE: Pause/Resume | S: Save | L: Load | T: Toggle Training/Inference
  Use UI sliders and buttons to adjust parameters
"""

# /// script
# [pygbag]
# autorun = true
# ///

import asyncio
import pygame
import sys
from typing import Tuple
from config import config, load_preset
from environments.snake import SnakeEnv
from core.q_learning import QLearningAgent
from ui.visualizer import RLVisualizer

class RLTrainer:
    """RL training application"""

    def __init__(self, load_model: bool = False):
        print("RLTrainer.__init__ starting...")

        # Create environment
        print("Creating SnakeEnv...")
        self.env = SnakeEnv(
            width=config.GAME_WIDTH,
            height=config.GAME_HEIGHT,
            block_size=config.BLOCK_SIZE
        )
        print(f"✓ SnakeEnv created: {self.env.width}x{self.env.height}")

        # Create agent
        print("Creating QLearningAgent...")
        self.agent = QLearningAgent(
            state_size=self.env.get_state_size(),
            action_size=self.env.get_action_size()
        )
        print("✓ QLearningAgent created")

        # Create visualizer
        print("Creating RLVisualizer...")
        self.visualizer = RLVisualizer(self.env, self.agent)
        print("✓ RLVisualizer created")

        # Training state
        self.running = True
        self.training = True
        self.inference_mode = False
        self.current_episode = 0
        self.demo_runs = 0

        # Time tracking for web (non-blocking)
        self.last_step_time = 0
        self.step_interval = 1.0 / config.GAME_SPEED

        print("=" * 70)
        print("Reinforcement Learning: Snake Q-Learning")
        print("=" * 70)
        print(f"Environment: {self.env.width}x{self.env.height} grid")
        print(f"State size: {self.env.get_state_size()}")
        print(f"Action size: {self.env.get_action_size()}")
        print(f"\nHyperparameters:")
        print(f"  Learning rate (α): {config.LEARNING_RATE}")
        print(f"  Discount (γ): {config.DISCOUNT_FACTOR}")
        print(f"  Epsilon start: {config.EPSILON_START}")
        print(f"  Epsilon decay: {config.EPSILON_DECAY}")
        print(f"\nTraining for {config.NUM_EPISODES} episodes")
        print("=" * 70)
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  S: Save model")
        print("  L: Load model")
        print("  T: Toggle Training/Inference mode")
        print("  Q: Quit")
        print("\nUI Buttons:")
        print("  - Adjust sliders and click Apply")
        print("  - Click preset buttons (Default, Fast, Slow, Turbo)")
        print("  - Save/Load Model buttons")
        print("  - Toggle Training mode button")
        print("=" * 70)

    def train_episode_step(self, state, current_time: float) -> Tuple[bool, float, float]:
        """
        Execute one step of training (non-blocking for web)
        Returns: (done, reward, next_state)
        """
        # Select action
        action = self.agent.get_action(state, training=not self.inference_mode)

        # Take step
        next_state, reward, done, info = self.env.step(action)

        # Learn
        if not self.inference_mode:
            self.agent.learn(state, action, reward, next_state, done)

        return next_state, reward, done

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            # Handle parameter panel events first
            panel_action = self.visualizer.handle_parameter_event(event)
            if panel_action == 'save_model':
                self.save_model()
            elif panel_action == 'load_model':
                self.load_model()
            elif panel_action == 'toggle_training':
                self.toggle_training_mode()

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.training = not self.training
                    print("Paused" if not self.training else "Resumed")
                elif event.key == pygame.K_s:
                    self.save_model()
                elif event.key == pygame.K_l:
                    self.load_model()
                elif event.key == pygame.K_t:
                    self.toggle_training_mode()
                elif event.key == pygame.K_q:
                    self.running = False

    def save_model(self, suffix: str = ''):
        """Save agent and stats"""
        print(f"\n⚠️ Save disabled in web version (browser storage not implemented)")
        print("  To save progress: use the desktop version (run_rl.py)")

    def load_model(self):
        """Load saved model"""
        print(f"\n⚠️ Load disabled in web version (browser storage not implemented)")
        print("  To load models: use the desktop version (run_rl.py)")

    def toggle_training_mode(self):
        """Toggle between training and inference mode"""
        self.inference_mode = not self.inference_mode

        if self.inference_mode:
            print("\n🔍 Inference Mode ENABLED")
            print("  - Agent will NOT learn")
            print("  - Epsilon = 0 (no exploration)")
            print("  - Watch trained agent perform!")
        else:
            print("\n🎓 Training Mode ENABLED")
            print("  - Agent will learn from experiences")
            print(f"  - Epsilon = {self.agent.epsilon:.3f} (exploration active)")
            print("  - Q-table will update")

        # Update UI
        self.visualizer.set_training_mode(not self.inference_mode)

    def update_speed_interval(self):
        """Update step interval based on config.GAME_SPEED"""
        if config.GAME_SPEED > 0:
            self.step_interval = 1.0 / config.GAME_SPEED


async def main():
    """Async main loop for web compatibility"""
    print("=== RL MODULE STARTING ===")
    # NOTE: pygame.init() is called automatically when RLVisualizer creates the display
    # Do NOT call pygame.init() here - it causes blank page in browser!

    try:
        print("Creating RLTrainer...")
        trainer = RLTrainer(load_model=False)
        print("✓ RLTrainer created")

        clock = pygame.time.Clock()
        print("✓ Clock created")
        print("=== READY! Training will start automatically ===")

        # Training loop
        best_score = 0
        episode = 0

        while trainer.running and episode < config.NUM_EPISODES:
            # Start new episode
            if not trainer.inference_mode:
                trainer.current_episode = episode
            else:
                trainer.demo_runs += 1

            state = trainer.env.reset()
            total_reward = 0
            trainer.visualizer.stats.start_episode()
            episode_done = False

            # Episode loop
            while trainer.running and not episode_done:
                current_time = pygame.time.get_ticks() / 1000.0

                # Handle events
                trainer.handle_events()

                # Update game state (time-based for web)
                if trainer.training:
                    if current_time - trainer.last_step_time >= trainer.step_interval:
                        # Execute one step
                        next_state, reward, done = trainer.train_episode_step(state, current_time)

                        total_reward += reward
                        trainer.visualizer.stats.record_step(reward)
                        state = next_state
                        episode_done = done

                        trainer.last_step_time = current_time

                        # Update step interval (in case speed changed)
                        trainer.update_speed_interval()

                # Render
                trainer.visualizer.render(
                    episode=trainer.current_episode,
                    current_state=state,
                    demo_runs=trainer.demo_runs
                )

                # CRITICAL: Yield control to browser
                await asyncio.sleep(0)
                clock.tick(config.FPS)

            # Episode complete
            if not trainer.inference_mode:
                trainer.agent.update_epsilon()
                trainer.visualizer.stats.end_episode(trainer.env.score)

                # Save best model
                if trainer.env.score > best_score:
                    best_score = trainer.env.score

                # Print progress
                if episode % 10 == 0:
                    summary = trainer.visualizer.stats.get_summary()
                    print(f"Episode {episode}/{config.NUM_EPISODES} | "
                          f"Score: {trainer.env.score} | "
                          f"Avg: {summary['avg_score']:.1f} | "
                          f"ε: {trainer.agent.epsilon:.3f}")
            else:
                # Inference mode
                if trainer.demo_runs % 5 == 0:
                    print(f"Demo Run {trainer.demo_runs} | "
                          f"Score: {trainer.env.score} | "
                          f"Mode: INFERENCE")

            episode += 1

        print(f"\n✓ Training complete! Best score: {best_score}")

    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Shutting down...")
        pygame.quit()


# Entry point - DON'T use asyncio.run() in browser!
# Pygbag will call main() directly
if __name__ == "__main__":
    asyncio.run(main())
