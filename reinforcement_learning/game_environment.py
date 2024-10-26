import pygame
import random
from config import GameConfig, Colors

class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        self.config = GameConfig()
        self.colors = Colors()
        
        # Display setup
        self.screen = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT))
        pygame.display.set_caption("Flappy Bird Q-Learning Demo")
        
        # Game elements
        self.reset()
        
        # Font
        self.font = pygame.font.Font(None, 36)

    def reset(self):
        """Resets the game state"""
        self.bird_pos = [self.config.WIDTH // 4, self.config.HEIGHT // 2]
        self.bird_velocity = 0
        self.pipes = []
        self.spawn_pipe()
        self.score = 0
        return self.get_state()

    def spawn_pipe(self):
        gap_y = random.randint(150, self.config.HEIGHT - 150)
        self.pipes.append({
            'x': self.config.WIDTH,
            'gap_y': gap_y,
            'passed': False
        })

    def get_state(self):
        if not self.pipes:
            return (0, 0, 0)
        
        pipe = self.pipes[0]
        dx = (pipe['x'] - self.bird_pos[0]) // 10
        dy = (pipe['gap_y'] - self.bird_pos[1]) // 10
        velocity_discrete = self.bird_velocity // 2
        
        return (int(dx), int(dy), int(velocity_discrete))

    def step(self, action):
        reward = 0.1  # Small reward for surviving
        done = False

        if action == 1:
            self.bird_velocity = self.config.FLAP_STRENGTH

        # Update bird position
        self.bird_velocity += self.config.GRAVITY
        self.bird_pos[1] += self.bird_velocity

        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= self.config.PIPE_SPEED
            
            if not pipe['passed'] and pipe['x'] < self.bird_pos[0]:
                pipe['passed'] = True
                self.score += 1
                reward = 1.0

        # Manage pipes
        if self.pipes and self.pipes[0]['x'] < -50:
            self.pipes.pop(0)
        if len(self.pipes) < 2:
            self.spawn_pipe()

        # Check collisions
        if self._check_collision():
            reward = -1000
            done = True

        return reward, self.get_state(), done

    def _check_collision(self):
        # Check pipe collisions
        for pipe in self.pipes:
            if (self.bird_pos[0] + 20 > pipe['x'] and
                self.bird_pos[0] - 20 < pipe['x'] + 50):
                if (self.bird_pos[1] - 20 < pipe['gap_y'] - 100 or
                    self.bird_pos[1] + 20 > pipe['gap_y'] + 100):
                    return True

        # Check boundaries
        if self.bird_pos[1] < 0 or self.bird_pos[1] > self.config.HEIGHT:
            return True
            
        return False

    def draw(self, mode="AI"):
        self.screen.fill(self.colors.BLACK)
        
        # Draw pipes
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, self.colors.GREEN,
                           (pipe['x'], 0, 50, pipe['gap_y'] - 100))
            pygame.draw.rect(self.screen, self.colors.GREEN,
                           (pipe['x'], pipe['gap_y'] + 100, 50, self.config.HEIGHT))
        
        # Draw bird
        bird_color = self.colors.RED if mode == "AI" else self.colors.BLUE
        pygame.draw.circle(self.screen, bird_color,
                         (int(self.bird_pos[0]), int(self.bird_pos[1])), 20)
        
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, self.colors.WHITE)
        mode_text = self.font.render(f'Mode: {mode}', True, self.colors.WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(mode_text, (10, 50))
        
        pygame.display.flip()