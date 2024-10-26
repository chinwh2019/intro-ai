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
        
        # Game specific parameters
        self.BIRD_RADIUS = 15
        self.PIPE_WIDTH = 70
        self.GAP_SIZE = 200     # Gap between top and bottom pipes
        
        # Game elements
        self.reset()
        
        # Font
        self.font = pygame.font.Font(None, 36)

    def spawn_pipe(self):
        """Creates a new pipe with a guaranteed passable gap"""
        # The gap_y represents the CENTER of the gap
        gap_y = random.randint(200, self.config.HEIGHT - 200)
        
        self.pipes.append({
            'x': self.config.WIDTH,
            'gap_y': gap_y,
            'passed': False
        })

    def _check_collision(self):
        """Simple collision detection"""
        for pipe in self.pipes:
            # Check if bird is at pipe's position
            if (pipe['x'] - self.BIRD_RADIUS < self.bird_pos[0] < pipe['x'] + self.PIPE_WIDTH + self.BIRD_RADIUS):
                # Check if bird hits the pipes
                if (self.bird_pos[1] < pipe['gap_y'] - self.GAP_SIZE/2 + self.BIRD_RADIUS or 
                    self.bird_pos[1] > pipe['gap_y'] + self.GAP_SIZE/2 - self.BIRD_RADIUS):
                    return True

        # Check if bird hits the ground or ceiling
        if self.bird_pos[1] < 0 or self.bird_pos[1] > self.config.HEIGHT:
            return True
            
        return False

    def draw(self, mode="AI"):
        """Clean drawing method"""
        self.screen.fill(self.colors.BLACK)
        
        # Draw pipes
        for pipe in self.pipes:
            # Top pipe
            pygame.draw.rect(
                self.screen, 
                self.colors.GREEN,
                (pipe['x'], 0, self.PIPE_WIDTH, pipe['gap_y'] - self.GAP_SIZE/2)
            )
            
            # Bottom pipe
            pygame.draw.rect(
                self.screen, 
                self.colors.GREEN,
                (pipe['x'], 
                 pipe['gap_y'] + self.GAP_SIZE/2,
                 self.PIPE_WIDTH,
                 self.config.HEIGHT - (pipe['gap_y'] + self.GAP_SIZE/2))
            )
        
        # Draw bird
        bird_color = self.colors.RED if mode == "AI" else self.colors.BLUE
        pygame.draw.circle(
            self.screen,
            bird_color,
            (int(self.bird_pos[0]), int(self.bird_pos[1])),
            self.BIRD_RADIUS
        )
        
        # Draw score and mode
        score_text = self.font.render(f'Score: {self.score}', True, self.colors.WHITE)
        mode_text = self.font.render(f'Mode: {mode}', True, self.colors.WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(mode_text, (10, 50))
        
        pygame.display.flip()

    def step(self, action):
        """Game step function"""
        reward = 0.1  # Small reward for staying alive
        done = False

        if action == 1:
            self.bird_velocity = self.config.FLAP_STRENGTH

        # Update bird position
        self.bird_velocity += self.config.GRAVITY
        self.bird_pos[1] += self.bird_velocity

        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= self.config.PIPE_SPEED
            
            # Award points for passing pipes
            if not pipe['passed'] and pipe['x'] + self.PIPE_WIDTH < self.bird_pos[0]:
                pipe['passed'] = True
                self.score += 1
                reward = 1.0

        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + self.PIPE_WIDTH > 0]

        # Add new pipe if needed
        if len(self.pipes) < 2:
            self.spawn_pipe()

        # Check for collisions
        if self._check_collision():
            reward = -1.0
            done = True

        return reward, self.get_state(), done

    def get_state(self):
        """Get current state for AI"""
        if not self.pipes:
            return (0, 0, 0)
        
        pipe = self.pipes[0]
        dx = (pipe['x'] - self.bird_pos[0]) / self.config.WIDTH
        dy = (pipe['gap_y'] - self.bird_pos[1]) / self.config.HEIGHT
        velocity = self.bird_velocity / 10
        
        return (int(dx * 10), int(dy * 10), int(velocity * 10))

    def reset(self):
        """Reset game state"""
        self.bird_pos = [self.config.WIDTH // 4, self.config.HEIGHT // 2]
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.spawn_pipe()
        return self.get_state()
