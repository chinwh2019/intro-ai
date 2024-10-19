import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

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
                V[s] = 1.0  # Set value of treasure state to 1
            elif s == mdp.trap:
                V[s] = -1.0  # Set value of trap state to -1
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
                    if a == "up":
                        s1 = (max(i-1, 0), j)
                    elif a == "down":
                        s1 = (min(i+1, grid_size-1), j)
                    elif a == "left":
                        s1 = (i, max(j-1, 0))
                    elif a == "right":
                        s1 = (i, min(j+1, grid_size-1))
                    
                    if s1 in obstacles:
                        s1 = s
                    
                    transition_probs[s][a][s1] = 1.0
    
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

def draw_treasure_trap_hunt(screen, grid_size, cell_size, start, treasure, trap, obstacles, policy=None, values=None):
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
            
            if policy and (i, j) in policy and (i, j) not in obstacles and (i, j) != treasure and (i, j) != trap:
                arrow = FONT.render(policy[(i, j)][0].upper(), True, RED)
                screen.blit(arrow, (rect.centerx - arrow.get_width() // 2, rect.centery - arrow.get_height() // 2))
            
            if values and (i, j) in values:
                value_text = FONT.render(f"{values[(i, j)]:.2f}", True, BLACK)
                screen.blit(value_text, (rect.x + 5, rect.y + 5))

def generate_random_positions(grid_size, num_obstacles):
    # Place treasure and trap close to each other
    treasure_x = random.randint(grid_size // 2, grid_size - 1)
    treasure_y = random.randint(grid_size // 2, grid_size - 1)
    treasure = (treasure_x, treasure_y)

    # Place trap adjacent to treasure
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    valid_traps = [(treasure_x + dx, treasure_y + dy) for dx, dy in directions 
                   if 0 <= treasure_x + dx < grid_size and 0 <= treasure_y + dy < grid_size]
    trap = random.choice(valid_traps)

    # Place start far from treasure and trap
    while True:
        start = (random.randint(0, grid_size // 2 - 1), random.randint(0, grid_size // 2 - 1))
        if (abs(start[0] - treasure[0]) + abs(start[1] - treasure[1])) > grid_size // 2:
            break

    # Generate obstacles
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
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Treasure and Trap MDP Value Iteration Visualization")

    num_obstacles = 8
    start, treasure, trap, obstacles = generate_random_positions(grid_size, num_obstacles)

    mdp = create_treasure_trap_mdp(grid_size, start, treasure, trap, obstacles)

    print("\nNow, let's apply the Value Iteration algorithm to find the optimal policy.")
    print("The algorithm will iterate until the values converge.")

    V, policy = value_iteration(mdp)

    running = True
    show_values = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    show_values = not show_values
                elif event.key == pygame.K_r:
                    start, treasure, trap, obstacles = generate_random_positions(grid_size, num_obstacles)
                    mdp = create_treasure_trap_mdp(grid_size, start, treasure, trap, obstacles)
                    V, policy = value_iteration(mdp)

        screen.fill(WHITE)
        draw_treasure_trap_hunt(screen, grid_size, cell_size, start, treasure, trap, obstacles, policy, V if show_values else None)
        
        instructions = FONT.render("Press SPACE to toggle state values, R to randomize", True, BLACK)
        screen.blit(instructions, (10, screen_size - 30))

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