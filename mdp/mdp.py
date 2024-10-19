import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)

# Font
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

def value_iteration(mdp, epsilon=0.001):
    V = {s: 0 for s in mdp.states}
    while True:
        delta = 0
        for s in mdp.states:
            if s == mdp.treasure:
                V[s] = 1.0
            elif s == mdp.trap:
                V[s] = -1.0
            else:
                v = V[s]
                V[s] = max(sum(mdp.transition_probs[s][a][s1] * (mdp.rewards[s][a] + mdp.discount * V[s1])
                               for s1 in mdp.states)
                           for a in mdp.actions)
                delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    
    policy = {}
    for s in mdp.states:
        if s == mdp.treasure or s == mdp.trap:
            policy[s] = "stay"
        else:
            policy[s] = max(mdp.actions,
                            key=lambda a: sum(mdp.transition_probs[s][a][s1] * (mdp.rewards[s][a] + mdp.discount * V[s1])
                                              for s1 in mdp.states))
    return V, policy

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
    
    return MDP(states, actions, transition_probs, rewards, discount=0.9, treasure=treasure, trap=trap)

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
        pygame.draw.polygon(screen, (255, 165, 0), points)  # Orange flames

def draw_start_symbol(screen, rect):
    pygame.draw.rect(screen, GREEN, rect)
    pygame.draw.polygon(screen, WHITE, [
        (rect.centerx, rect.top + rect.height * 0.2),
        (rect.centerx - rect.width * 0.3, rect.bottom - rect.height * 0.2),
        (rect.centerx + rect.width * 0.3, rect.bottom - rect.height * 0.2)
    ])

def draw_treasure_trap_hunt(screen, grid_size, cell_size, start, treasure, trap, obstacles, V, policy=None, show_policy=False):
    for i in range(grid_size):
        for j in range(grid_size):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            if (i, j) in obstacles:
                pygame.draw.rect(screen, GRAY, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
            
            if (i, j) == start:
                pygame.draw.circle(screen, BLUE, rect.center, cell_size // 3)
            elif (i, j) == treasure:
                draw_start_symbol(screen, rect)
            elif (i, j) == trap:
                draw_fire_pit(screen, rect)
            
            value_text = FONT.render(f"{V[(i, j)]:.2f}", True, BLACK)
            screen.blit(value_text, (rect.x + 5, rect.y + 5))
            
            if show_policy and policy and (i, j) in policy and (i, j) not in obstacles and (i, j) != treasure and (i, j) != trap:
                arrow = FONT.render(policy[(i, j)][0].upper(), True, RED)
                screen.blit(arrow, (rect.centerx - arrow.get_width() // 2, rect.centery - arrow.get_height() // 2))


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

def main():
    print("Welcome to the Treasure and Trap MDP and Value Iteration Algorithm Explainer!")
    print("This application will guide you through the concepts using a treasure hunt game with a trap.")
    
    grid_size = 8
    cell_size = 80
    screen_size = grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size + 60))  # Increased extra space for two lines of text
    pygame.display.set_caption("Treasure and Trap MDP Value Iteration Visualization")

    num_obstacles = 8
    start, treasure, trap, obstacles = generate_random_positions(grid_size, num_obstacles)

    mdp = create_treasure_trap_mdp(grid_size, start, treasure, trap, obstacles)

    print("\nNow, let's apply the Value Iteration algorithm to find the optimal policy.")
    print("The algorithm will iterate until the values converge.")

    V, policy = value_iteration(mdp)

    running = True
    show_policy = False

    # Create a button
    button_rect = pygame.Rect(10, screen_size + 15, 150, 30)
    button_color = LIGHT_BLUE
    button_text = FONT.render("Toggle Policy", True, BLACK)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    show_policy = not show_policy
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    show_policy = not show_policy
                elif event.key == pygame.K_r:
                    start, treasure, trap, obstacles = generate_random_positions(grid_size, num_obstacles)
                    mdp = create_treasure_trap_mdp(grid_size, start, treasure, trap, obstacles)
                    V, policy = value_iteration(mdp)

        screen.fill(WHITE)
        draw_treasure_trap_hunt(screen, grid_size, cell_size, start, treasure, trap, obstacles, V, policy, show_policy)
        
        # Draw button
        pygame.draw.rect(screen, button_color, button_rect)
        screen.blit(button_text, (button_rect.x + 10, button_rect.y + 5))

        # Split instructions into two lines
        instructions1 = FONT.render("SPACE or click button to toggle policy", True, BLACK)
        instructions2 = FONT.render("Press R to randomize", True, BLACK)
        screen.blit(instructions1, (170, screen_size + 10))
        screen.blit(instructions2, (170, screen_size + 35))

        pygame.display.flip()

    pygame.quit()

    print("\nKey Concepts:")
    print("- States: Each cell in the grid")
    print("- Actions: Up, Down, Left, Right")
    print("- Rewards: -0.04 for each move, -0.1 for obstacles, 1.0 for treasure, -1.0 for trap")
    print("- Discount factor: 0.9 (future rewards are slightly less valuable)")
    print("- Policy: The best action to take in each state")

    print("\nThank you for using the Treasure and Trap MDP and Value Iteration Explainer!")

if __name__ == "__main__":
    main()