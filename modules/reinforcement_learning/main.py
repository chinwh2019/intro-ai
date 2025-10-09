"""Main training loop for RL"""

import pygame
import sys
import time
import os
from typing import Tuple
from modules.reinforcement_learning.config import config
from modules.reinforcement_learning.environments.snake import SnakeEnv
from modules.reinforcement_learning.core.q_learning import QLearningAgent
from modules.reinforcement_learning.ui.visualizer import RLVisualizer

class RLTrainer:
    """RL training application"""

    def __init__(self, load_model: bool = False):
        # Create environment
        self.env = SnakeEnv(
            width=config.GAME_WIDTH,
            height=config.GAME_HEIGHT,
            block_size=config.BLOCK_SIZE
        )

        # Create agent
        self.agent = QLearningAgent(
            state_size=self.env.get_state_size(),
            action_size=self.env.get_action_size()
        )

        # Load model if requested
        if load_model:
            self.agent.load(config.MODEL_SAVE_PATH)

        # Create visualizer
        self.visualizer = RLVisualizer(self.env, self.agent)

        # Training state
        self.running = True
        self.training = True
        self.inference_mode = False  # When True, agent doesn't learn, epsilon=0
        self.current_episode = 0

        # Ensure model directory exists
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

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
        print("  - Click preset buttons (Default, Fast, Slow, Demo)")
        print("  - Save/Load Model buttons")
        print("  - Toggle Training mode button")
        print("=" * 70)

    def train_episode(self) -> Tuple[int, float]:
        """Train one episode"""
        state = self.env.reset()
        total_reward = 0
        self.visualizer.stats.start_episode()

        while True:
            # Handle events
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
                    return self.env.score, total_reward

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
                        return self.env.score, total_reward

            if not self.training:
                time.sleep(0.1)
                self.visualizer.render(
                    episode=self.current_episode,
                    current_state=state
                )
                continue

            # Select action (use inference mode if enabled)
            action = self.agent.get_action(state, training=not self.inference_mode)

            # Take step
            next_state, reward, done, info = self.env.step(action)

            # Learn (only if not in inference mode)
            if not self.inference_mode:
                self.agent.learn(state, action, reward, next_state, done)

            # Update visualization
            self.visualizer.render(
                episode=self.current_episode,
                current_state=state
            )

            # Control game speed (steps per second)
            if config.GAME_SPEED > 0:
                time.sleep(1.0 / config.GAME_SPEED)

            # Update stats
            total_reward += reward
            self.visualizer.stats.record_step(reward)

            # Next state
            state = next_state

            if done:
                break

        return self.env.score, total_reward

    def train(self):
        """Main training loop"""
        best_score = 0

        for episode in range(config.NUM_EPISODES):
            if not self.running:
                break

            self.current_episode = episode

            # Train episode
            score, total_reward = self.train_episode()

            # Update epsilon (only if training)
            if not self.inference_mode:
                self.agent.update_epsilon()

            # Record stats
            self.visualizer.stats.end_episode(score)

            # Save best model
            if score > best_score:
                best_score = score
                self.save_model(suffix='_best')

            # Print progress
            if episode % 10 == 0:
                summary = self.visualizer.stats.get_summary()
                print(f"Episode {episode}/{config.NUM_EPISODES} | "
                      f"Score: {score} | "
                      f"Avg Score: {summary['avg_score']:.1f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"Best: {best_score}")

            # Periodic save
            if episode % 100 == 0 and episode > 0:
                self.save_model()

        # Final save
        self.save_model()
        print(f"\n✓ Training complete!")
        print(f"  Best score: {best_score}")
        print(f"  Final avg score: {self.visualizer.stats.get_summary()['avg_score']:.1f}")

    def save_model(self, suffix: str = ''):
        """Save agent and stats"""
        model_path = config.MODEL_SAVE_PATH
        if suffix:
            model_path = model_path.replace('.json', f'{suffix}.json')

        self.agent.save(model_path)
        self.visualizer.stats.save(config.STATS_SAVE_PATH)
        print(f"✓ Model saved to {model_path}")

    def load_model(self):
        """Load saved model"""
        if self.agent.load(config.MODEL_SAVE_PATH):
            print("\n✓ Model loaded successfully!")
            print(f"  Q-table size: {len(self.agent.q_table)}")
            print(f"  Episodes trained: {self.agent.episodes_trained}")
            print(f"  Current epsilon: {self.agent.epsilon:.3f}")
        else:
            print("\n✗ Failed to load model")
            print(f"  File: {config.MODEL_SAVE_PATH}")
            print("  Train and save a model first!")

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

    def run(self):
        """Run training"""
        try:
            self.train()
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self.save_model()
        finally:
            pygame.quit()
            sys.exit()


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Snake Q-Learning')
    parser.add_argument('--load', action='store_true', help='Load existing model')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes')
    args = parser.parse_args()

    if args.episodes:
        config.NUM_EPISODES = args.episodes

    trainer = RLTrainer(load_model=args.load)
    trainer.run()


if __name__ == '__main__':
    main()
