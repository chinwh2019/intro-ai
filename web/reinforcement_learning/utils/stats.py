"""Training statistics tracking - Browser-safe (no NumPy)"""

from typing import List

class TrainingStats:
    """Track training statistics"""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_scores: List[int] = []
        self.episode_lengths: List[int] = []
        self.avg_rewards: List[float] = []
        self.avg_scores: List[float] = []

        # Current episode stats
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def start_episode(self):
        """Start new episode"""
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def record_step(self, reward: float):
        """Record step in current episode"""
        self.current_episode_reward += reward
        self.current_episode_length += 1

    def end_episode(self, score: int):
        """End episode and record stats"""
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_scores.append(score)
        self.episode_lengths.append(self.current_episode_length)

        # Calculate moving averages (last 100 episodes) - pure Python
        window = 100
        if len(self.episode_rewards) >= window:
            recent_rewards = self.episode_rewards[-window:]
            self.avg_rewards.append(sum(recent_rewards) / len(recent_rewards))

            recent_scores = self.episode_scores[-window:]
            self.avg_scores.append(sum(recent_scores) / len(recent_scores))

    def get_summary(self, last_n: int = 100) -> dict:
        """Get summary statistics - pure Python"""
        recent_scores = self.episode_scores[-last_n:]
        recent_rewards = self.episode_rewards[-last_n:]

        return {
            'total_episodes': len(self.episode_scores),
            'avg_score': (sum(recent_scores) / len(recent_scores)) if recent_scores else 0.0,
            'max_score': max(recent_scores) if recent_scores else 0,
            'min_score': min(recent_scores) if recent_scores else 0,
            'avg_reward': (sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0,
        }

    def save(self, filepath: str):
        """Save statistics"""
        import json
        import os

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'avg_rewards': self.avg_rewards,
            'avg_scores': self.avg_scores,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load statistics"""
        import json
        import os

        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_scores = data.get('episode_scores', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.avg_rewards = data.get('avg_rewards', [])
        self.avg_scores = data.get('avg_scores', [])

        return True
