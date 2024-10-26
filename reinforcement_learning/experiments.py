import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from config import ExperimentConfig
from utils import PathManager
import time


class ExperimentManager:
    def __init__(self, game, agent):
        self.game = game
        self.agent = agent
        self.experiment_scores = []
        self.running_reward = deque(maxlen=100)
        self.results_dir = PathManager.get_data_dir() / 'results'
        self.results_dir.mkdir(exist_ok=True)
    
    def save_experiment_results(self, results, experiment_name):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = self.results_dir / f'{experiment_name}_{timestamp}.png'
        plt.savefig(filename)
        plt.close()

    def plot_results(self, results, title):
        plt.figure(figsize=(10, 6))
        for key, scores in results.items():
            plt.plot(scores, label=f'Parameter: {key}')
        plt.title(title)
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

    def run_training_episodes(self, episodes, config):
        scores = []
        self.agent.learning_rate = config.learning_rate
        self.agent.discount_factor = config.discount_factor
        self.agent.epsilon = config.epsilon
        
        for episode in range(episodes):
            state = self.game.reset()
            episode_score = 0
            done = False
            
            while not done:
                action = self.agent.get_action(state)
                reward, next_state, done = self.game.step(action)
                self.agent.update(state, action, reward, next_state)
                state = next_state
                episode_score += reward
                
                self.game.draw("AI")  # Update display
                
            self.agent.epsilon = max(
                config.min_epsilon,
                self.agent.epsilon * config.epsilon_decay
            )
            scores.append(episode_score)
            print(f"Episode {episode + 1}/{episodes}, Score: {episode_score}")
            
        return scores

    def learning_rate_experiment(self):
        learning_rates = [0.01, 0.1, 0.5]
        results = {}
        
        for lr in learning_rates:
            config = ExperimentConfig(
                learning_rate=lr,
                discount_factor=0.95,
                epsilon=0.1,
                epsilon_decay=0.995,
                min_epsilon=0.01
            )
            scores = self.run_training_episodes(100, config)
            results[lr] = scores
            
        self.plot_results(results, "Learning Rate Comparison")
