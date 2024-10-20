import pygame
import sys
import random
import math

# Constants
GRID_SIZE = 5
CELL_SIZE = 120
NUM_OBSTACLES = 2
LEARNING_RATE = 0.1
DISCOUNT = 0.9
MAX_EPISODES = 1000
DEMO_STEP_DELAY = 500

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
DARK_GREEN = (0, 100, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
GOLD = (255, 215, 0)

# Initialize Pygame
pygame.init()
FONT = pygame.font.Font(None, 24)

class MDP:
    def __init__(self, states, actions, transition_probs, rewards, discount, treasure, trap):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.discount = discount
        self.treasure = treasure
        self.trap = trap

class MDPSolver:
    @staticmethod
    def value_iteration(mdp, epsilon=0.001):
        V = {s: 0 for s in mdp.states}
        iterations = []
        
        while True:
            delta = 0
            Q = {s: {a: 0 for a in mdp.actions} for s in mdp.states}
            for s in mdp.states:
                if s == mdp.treasure:
                    V[s] = 1.0
                    Q[s] = {a: 1.0 for a in mdp.actions}
                elif s == mdp.trap:
                    V[s] = -1.0
                    Q[s] = {a: -1.0 for a in mdp.actions}
                else:
                    v = V[s]
                    for a in mdp.actions:
                        Q[s][a] = sum(mdp.transition_probs[s][a][s1] * (mdp.rewards[s][a] + mdp.discount * V[s1])
                                      for s1 in mdp.states)
                    V[s] = max(Q[s].values())
                    delta = max(delta, abs(v - V[s]))
            
            iterations.append((V.copy(), Q.copy()))
            if delta < epsilon:
                break
        
        policy = {s: max(Q[s], key=Q[s].get) for s in mdp.states}
        return V, Q, policy, iterations

class MDPEnvironment:
    @staticmethod
    def create_treasure_trap_mdp(grid_size, start, treasure, trap, obstacles):
        states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        actions = ["up", "down", "left", "right"]
        
        transition_probs = {s: {a: {s1: 0.0 for s1 in states} for a in actions} for s in states}
        rewards = {s: {a: -0.04 for a in actions} for s in states}
        
        p_intended = 0.8
        p_perpendicular = 0.1
        
        for i in range(grid_size):
            for j in range(grid_size):
                s = (i, j)
                for a in actions:
                    if s == treasure:
                        transition_probs[s][a][s] = 1.0
                        rewards[s][a] = 1.0
                    elif s == trap:
                        transition_probs[s][a][s] = 1.0
                        rewards[s][a] = -1.0
                    elif s in obstacles:
                        transition_probs[s][a][s] = 1.0
                        rewards[s][a] = -0.1
                    else:
                        for a_actual in actions:
                            if a == a_actual:
                                p = p_intended
                            elif (a in ["up", "down"] and a_actual in ["left", "right"]) or \
                                 (a in ["left", "right"] and a_actual in ["up", "down"]):
                                p = p_perpendicular
                            else:
                                p = 0
                            
                            if a_actual == "up":
                                s1 = (max(i-1, 0), j)
                            elif a_actual == "down":
                                s1 = (min(i+1, grid_size-1), j)
                            elif a_actual == "left":
                                s1 = (i, max(j-1, 0))
                            elif a_actual == "right":
                                s1 = (i, min(j+1, grid_size-1))
                            
                            if s1 in obstacles:
                                s1 = s
                            
                            transition_probs[s][a][s1] += p
        
        return MDP(states, actions, transition_probs, rewards, discount=DISCOUNT, treasure=treasure, trap=trap)

    @staticmethod
    def generate_random_positions(grid_size, num_obstacles):
        treasure_x = random.randint(grid_size // 2, grid_size - 1)
        treasure_y = random.randint(grid_size // 2, grid_size - 1)
        treasure = (treasure_x, treasure_y)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        valid_traps = [(treasure_x + dx, treasure_y + dy) for dx, dy in directions 
                       if 0 <= treasure_x + dx < grid_size and 0 <= treasure_y + dy < grid_size]
        trap = random.choice(valid_traps)

        while True:
            start = (random.randint(0, grid_size // 2 - 1), random.randint(0, grid_size // 2 - 1))
            if (abs(start[0] - treasure[0]) + abs(start[1] - treasure[1])) > grid_size // 2:
                break

        all_positions = set((i, j) for i in range(grid_size) for j in range(grid_size))
        all_positions -= {start, treasure, trap}
        obstacles = random.sample(list(all_positions), min(num_obstacles, len(all_positions)))

        return start, treasure, trap, obstacles

class MDPLearner:
    @staticmethod
    def update_values(state, action, next_state, reward, V, Q, discount, learning_rate):
        Q[state][action] = (1 - learning_rate) * Q[state][action] + \
                           learning_rate * (reward + discount * max(Q[next_state].values()))
        V[state] = max(Q[state].values())

    @staticmethod
    def follow_policy(state, policy, mdp):
        action = policy[state]
        next_states = list(mdp.transition_probs[state][action].keys())
        probabilities = list(mdp.transition_probs[state][action].values())
        next_state = random.choices(next_states, weights=probabilities, k=1)[0]
        return next_state

class MDPVisualizer:
    @staticmethod
    def draw_fire_pit(screen, rect):
        pygame.draw.rect(screen, RED, rect)
        flame_height = rect.height // 2
        flame_width = rect.width // 4
        for i in range(4):
            points = [
                (rect.left + i * flame_width, rect.bottom),
                (rect.left + (i + 0.5) * flame_width, rect.bottom - flame_height),
                (rect.left + (i + 1) * flame_width, rect.bottom)
            ]
            pygame.draw.polygon(screen, (255, 165, 0), points)

    @staticmethod
    def draw_start_symbol(screen, rect):
        pygame.draw.rect(screen, GREEN, rect)
        pygame.draw.polygon(screen, WHITE, [
            (rect.centerx, rect.top + rect.height * 0.2),
            (rect.centerx - rect.width * 0.3, rect.bottom - rect.height * 0.2),
            (rect.centerx + rect.width * 0.3, rect.bottom - rect.height * 0.2)
        ])

    @staticmethod
    def draw_triangle(surface, color, points):
        pygame.draw.polygon(surface, color, points)
        pygame.draw.lines(surface, WHITE, True, points, 1)

    @staticmethod
    def draw_cell(surface, rect, state_value, q_values, show_q_values):
        pygame.draw.rect(surface, DARK_GREEN, rect)
        pygame.draw.rect(surface, WHITE, rect, 1)

        value_text = FONT.render(f"{state_value:.2f}", True, WHITE)
        surface.blit(value_text, (rect.centerx - value_text.get_width() // 2, rect.centery - value_text.get_height() // 2))

        if show_q_values:
            triangle_points = [
                [(rect.left, rect.top), (rect.right, rect.top), (rect.centerx, rect.centery)],
                [(rect.right, rect.top), (rect.right, rect.bottom), (rect.centerx, rect.centery)],
                [(rect.left, rect.bottom), (rect.right, rect.bottom), (rect.centerx, rect.centery)],
                [(rect.left, rect.top), (rect.left, rect.bottom), (rect.centerx, rect.centery)]
            ]
            
            for i, direction in enumerate(['up', 'right', 'down', 'left']):
                q_value = q_values[direction]
                color = RED if q_value < 0 else DARK_GREEN
                MDPVisualizer.draw_triangle(surface, color, triangle_points[i])
                q_text = FONT.render(f"{q_value:.2f}", True, WHITE)
                
                if direction == 'up':
                    text_pos = (rect.centerx - q_text.get_width() // 2, rect.top + 5)
                elif direction == 'right':
                    text_pos = (rect.right - q_text.get_width() - 5, rect.centery - q_text.get_height() // 2)
                elif direction == 'down':
                    text_pos = (rect.centerx - q_text.get_width() // 2, rect.bottom - q_text.get_height() - 5)
                else:  # left
                    text_pos = (rect.left + 5, rect.centery - q_text.get_height() // 2)
                
                surface.blit(q_text, text_pos)

    @staticmethod
    def draw_treasure_trap_hunt(screen, grid_size, cell_size, start, treasure, trap, obstacles, V, Q, policy, show_policy, show_q_values, walker_pos=None):
        for i in range(grid_size):
            for j in range(grid_size):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                
                if (i, j) in obstacles:
                    pygame.draw.rect(screen, GRAY, rect)
                elif (i, j) == trap:
                    pygame.draw.rect(screen, RED, rect)
                    value_text = FONT.render(f"{V[(i, j)]:.2f}", True, WHITE)
                    screen.blit(value_text, (rect.centerx - value_text.get_width() // 2, rect.centery - value_text.get_height() // 2))
                elif (i, j) == treasure:
                    pygame.draw.rect(screen, GOLD, rect)
                    value_text = FONT.render(f"{V[(i, j)]:.2f}", True, BLACK)
                    screen.blit(value_text, (rect.centerx - value_text.get_width() // 2, rect.centery - value_text.get_height() // 2))
                else:
                    MDPVisualizer.draw_cell(screen, rect, V[(i, j)], Q[(i, j)], show_q_values)
                
                pygame.draw.rect(screen, WHITE, rect, 1)
                
                if (i, j) == start:
                    pygame.draw.circle(screen, BLUE, rect.center, cell_size // 4)
                
                if show_policy and policy and (i, j) in policy and (i, j) not in obstacles and (i, j) != treasure and (i, j) != trap:
                    arrow_points = {
                        'up': [(rect.centerx, rect.top + 5), (rect.centerx - 5, rect.top + 15), (rect.centerx + 5, rect.top + 15)],
                        'down': [(rect.centerx, rect.bottom - 5), (rect.centerx - 5, rect.bottom - 15), (rect.centerx + 5, rect.bottom - 15)],
                        'left': [(rect.left + 5, rect.centery), (rect.left + 15, rect.centery - 5), (rect.left + 15, rect.centery + 5)],
                        'right': [(rect.right - 5, rect.centery), (rect.right - 15, rect.centery - 5), (rect.right - 15, rect.centery + 5)]
                    }
                    pygame.draw.polygon(screen, WHITE, arrow_points[policy[(i, j)]])
                
                if walker_pos and (i, j) == walker_pos:
                    pygame.draw.circle(screen, (255, 165, 0), rect.center, cell_size // 3)

class MDPSimulation:
    def __init__(self):
        self.screen_size = GRID_SIZE * CELL_SIZE
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size + 200))
        pygame.display.set_caption("Treasure and Trap MDP Visualization")

        self.start, self.treasure, self.trap, self.obstacles = MDPEnvironment.generate_random_positions(GRID_SIZE, NUM_OBSTACLES)
        self.mdp = MDPEnvironment.create_treasure_trap_mdp(GRID_SIZE, self.start, self.treasure, self.trap, self.obstacles)
        self.V, self.Q, self.policy, self.iterations = MDPSolver.value_iteration(self.mdp)

        self.show_policy = False
        self.show_q_values = False
        self.show_learning_process = False
        self.manual_learning_mode = False
        self.policy_demo_mode = False
        self.walker_pos = self.start

        self.episode_count = 0
        self.current_iteration = 0
        self.terminal_state_message = ""

        self.setup_buttons()

    def setup_buttons(self):
        button_height = 30
        button_spacing = 10
        button_y1 = self.screen_size + 15
        button_y2 = button_y1 + button_height + button_spacing

        self.policy_button_rect = pygame.Rect(10, button_y1, 150, button_height)
        self.q_value_button_rect = pygame.Rect(170, button_y1, 150, button_height)
        self.learning_button_rect = pygame.Rect(330, button_y1, 150, button_height)
        self.manual_learning_button_rect = pygame.Rect(10, button_y2, 150, button_height)
        self.policy_demo_button_rect = pygame.Rect(170, button_y2, 150, button_height)

        self.button_color = LIGHT_GRAY
        self.policy_button_text = FONT.render("Toggle Policy", True, BLACK)
        self.q_value_button_text = FONT.render("Toggle Q-Values", True, BLACK)
        self.learning_button_text = FONT.render("Show Learning", True, BLACK)
        self.manual_learning_text = FONT.render("Manual Learning", True, BLACK)
        self.policy_demo_text = FONT.render("Policy Demo", True, BLACK)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)
            if event.type == pygame.KEYDOWN:
                self.handle_key_press(event.key)
        return True

    def handle_mouse_click(self, pos):
        if self.policy_button_rect.collidepoint(pos):
            self.show_policy = not self.show_policy
        elif self.q_value_button_rect.collidepoint(pos):
            self.show_q_values = not self.show_q_values
        elif self.learning_button_rect.collidepoint(pos):
            self.show_learning_process = not self.show_learning_process
            self.current_iteration = 0
        elif self.manual_learning_button_rect.collidepoint(pos):
            self.toggle_manual_learning()
        elif self.policy_demo_button_rect.collidepoint(pos):
            self.toggle_policy_demo()

    def handle_key_press(self, key):
        if key == pygame.K_SPACE:
            self.show_policy = not self.show_policy
        elif key == pygame.K_q:
            self.show_q_values = not self.show_q_values
        elif key == pygame.K_l:
            self.show_learning_process = not self.show_learning_process
            self.current_iteration = 0
        elif key == pygame.K_r:
            self.reset_environment()
        elif self.manual_learning_mode and self.episode_count < MAX_EPISODES:
            self.handle_manual_learning(key)

    def toggle_manual_learning(self):
        self.manual_learning_mode = not self.manual_learning_mode
        self.policy_demo_mode = False
        if self.manual_learning_mode:
            self.reset_learning()

    def toggle_policy_demo(self):
        self.policy_demo_mode = not self.policy_demo_mode
        self.manual_learning_mode = False
        if self.policy_demo_mode:
            self.walker_pos = random.choice([s for s in self.mdp.states if s not in self.obstacles and s != self.treasure and s != self.trap])

    def reset_environment(self):
        self.start, self.treasure, self.trap, self.obstacles = MDPEnvironment.generate_random_positions(GRID_SIZE, NUM_OBSTACLES)
        self.mdp = MDPEnvironment.create_treasure_trap_mdp(GRID_SIZE, self.start, self.treasure, self.trap, self.obstacles)
        self.V, self.Q, self.policy, self.iterations = MDPSolver.value_iteration(self.mdp)
        self.walker_pos = self.start
        self.episode_count = 0

    def reset_learning(self):
        self.V = {s: 0 for s in self.mdp.states}
        self.Q = {s: {a: 0 for a in self.mdp.actions} for s in self.mdp.states}
        self.V[self.treasure] = 1.0
        self.V[self.trap] = -1.0
        for a in self.mdp.actions:
            self.Q[self.treasure][a] = 1.0
            self.Q[self.trap][a] = -1.0
        self.walker_pos = self.start
        self.episode_count = 0

    def handle_manual_learning(self, key):
        action_map = {pygame.K_UP: "up", pygame.K_DOWN: "down", pygame.K_LEFT: "left", pygame.K_RIGHT: "right"}
        if key in action_map:
            action = action_map[key]
            next_state = MDPLearner.follow_policy(self.walker_pos, {self.walker_pos: action}, self.mdp)
            reward = self.mdp.rewards[self.walker_pos][action]
            MDPLearner.update_values(self.walker_pos, action, next_state, reward, self.V, self.Q, DISCOUNT, LEARNING_RATE)
            self.walker_pos = next_state

            if self.walker_pos in [self.treasure, self.trap]:
                self.episode_count += 1
                self.walker_pos = self.start
                print(f"Episode {self.episode_count} completed")

    def update(self):
        if self.show_learning_process and self.current_iteration < len(self.iterations):
            V_current, Q_current = self.iterations[self.current_iteration]
            MDPVisualizer.draw_treasure_trap_hunt(self.screen, GRID_SIZE, CELL_SIZE, self.start, self.treasure, self.trap, 
                                                  self.obstacles, V_current, Q_current, self.policy, self.show_policy, 
                                                  self.show_q_values, self.walker_pos)
            self.current_iteration += 1
            if self.current_iteration == len(self.iterations):
                self.show_learning_process = False
            pygame.time.wait(200)
        else:
            MDPVisualizer.draw_treasure_trap_hunt(self.screen, GRID_SIZE, CELL_SIZE, self.start, self.treasure, self.trap, 
                                                  self.obstacles, self.V, self.Q, self.policy, self.show_policy, 
                                                  self.show_q_values, self.walker_pos)

        if self.policy_demo_mode:
            self.update_policy_demo()

        self.draw_ui()

    def update_policy_demo(self):
        if self.walker_pos not in [self.treasure, self.trap]:
            self.terminal_state_message = ""
            pygame.time.wait(DEMO_STEP_DELAY)
            self.walker_pos = MDPLearner.follow_policy(self.walker_pos, self.policy, self.mdp)
            if self.walker_pos == self.treasure:
                self.terminal_state_message = "Reached the Treasure! Episode Ended."
                pygame.time.wait(1000)
                self.walker_pos = random.choice([s for s in self.mdp.states if s not in self.obstacles and s != self.treasure and s != self.trap])
            elif self.walker_pos == self.trap:
                self.terminal_state_message = "Fell into the Trap! Episode Ended."
                pygame.time.wait(1000)
                self.walker_pos = random.choice([s for s in self.mdp.states if s not in self.obstacles and s != self.treasure and s != self.trap])

    def draw_ui(self):
        pygame.draw.rect(self.screen, self.button_color, self.policy_button_rect)
        pygame.draw.rect(self.screen, self.button_color, self.q_value_button_rect)
        pygame.draw.rect(self.screen, self.button_color, self.learning_button_rect)
        pygame.draw.rect(self.screen, self.button_color, self.manual_learning_button_rect)
        pygame.draw.rect(self.screen, self.button_color, self.policy_demo_button_rect)
        
        self.screen.blit(self.policy_button_text, (self.policy_button_rect.x + 10, self.policy_button_rect.y + 5))
        self.screen.blit(self.q_value_button_text, (self.q_value_button_rect.x + 10, self.q_value_button_rect.y + 5))
        self.screen.blit(self.learning_button_text, (self.learning_button_rect.x + 10, self.learning_button_rect.y + 5))
        self.screen.blit(self.manual_learning_text, (self.manual_learning_button_rect.x + 10, self.manual_learning_button_rect.y + 5))
        self.screen.blit(self.policy_demo_text, (self.policy_demo_button_rect.x + 10, self.policy_demo_button_rect.y + 5))

        text_y = self.policy_demo_button_rect.bottom + 20
        if self.terminal_state_message:
            message_text = FONT.render(self.terminal_state_message, True, RED)
            self.screen.blit(message_text, (10, text_y))
            text_y += 30

        mode_text = FONT.render(f"Mode: {'Policy Demo' if self.policy_demo_mode else 'Manual Learning' if self.manual_learning_mode else 'Observation'}", True, BLACK)
        self.screen.blit(mode_text, (10, text_y))
        text_y += 30

        instructions = FONT.render("SPACE/P: Policy, Q: Q-values, L: learning, R: Randomize maze", True, BLACK)
        self.screen.blit(instructions, (10, text_y))
        instructions = FONT.render("Arrow keys: Manual move", True, BLACK)
        self.screen.blit(instructions, (10, text_y + 20))

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            running = self.handle_events()
            self.screen.fill(WHITE)
            self.update()
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

def main():
    simulation = MDPSimulation()
    simulation.run()

    print("\nKey Concepts:")
    print("- States: Each cell in the grid")
    print("- Actions: Up, Down, Left, Right")
    print("- Rewards: -0.04 for each move, -0.1 for obstacles, 1.0 for treasure, -1.0 for trap")
    print(f"- Discount factor: {DISCOUNT} (future rewards are slightly less valuable)")
    print("- Policy: The best action to take in each state")

if __name__ == "__main__":
    main()