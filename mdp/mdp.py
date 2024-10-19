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
DARK_GREEN = (0, 100, 0)  # Darker green color
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
GOLD = (255, 215, 0)  # Gold color for the treasure

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


def draw_triangle(surface, color, points):
    pygame.draw.polygon(surface, color, points)
    pygame.draw.lines(surface, WHITE, True, points, 1)


def draw_cell(surface, rect, state_value, q_values, show_q_values):
    pygame.draw.rect(surface, DARK_GREEN, rect)
    pygame.draw.rect(surface, WHITE, rect, 1)

    # Draw state value
    value_text = FONT.render(f"{state_value:.2f}", True, WHITE)
    surface.blit(value_text, (rect.centerx - value_text.get_width() // 2, rect.centery - value_text.get_height() // 2))

    if show_q_values:
        # Draw Q-values in triangles
        triangle_points = [
            [(rect.left, rect.top), (rect.right, rect.top), (rect.centerx, rect.centery)],  # Up
            [(rect.right, rect.top), (rect.right, rect.bottom), (rect.centerx, rect.centery)],  # Right
            [(rect.left, rect.bottom), (rect.right, rect.bottom), (rect.centerx, rect.centery)],  # Down
            [(rect.left, rect.top), (rect.left, rect.bottom), (rect.centerx, rect.centery)]  # Left
        ]
        
        for i, direction in enumerate(['up', 'right', 'down', 'left']):
            q_value = q_values[direction]
            color = RED if q_value < 0 else DARK_GREEN
            draw_triangle(surface, color, triangle_points[i])
            q_text = FONT.render(f"{q_value:.2f}", True, WHITE)
            
            # Calculate text position
            if direction == 'up':
                text_pos = (rect.centerx - q_text.get_width() // 2, rect.top + 5)
            elif direction == 'right':
                text_pos = (rect.right - q_text.get_width() - 5, rect.centery - q_text.get_height() // 2)
            elif direction == 'down':
                text_pos = (rect.centerx - q_text.get_width() // 2, rect.bottom - q_text.get_height() - 5)
            else:  # left
                text_pos = (rect.left + 5, rect.centery - q_text.get_height() // 2)
            
            surface.blit(q_text, text_pos)


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
                draw_cell(screen, rect, V[(i, j)], Q[(i, j)], show_q_values)
            
            # Draw border for all cells, including treasure and trap
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
            
            # Draw walker
            if walker_pos and (i, j) == walker_pos:
                pygame.draw.circle(screen, (255, 165, 0), rect.center, cell_size // 3)  # Orange circle for walker


def update_values(state, action, next_state, reward, V, Q, discount, learning_rate):
    # Q-learning update
    Q[state][action] = (1 - learning_rate) * Q[state][action] + \
                       learning_rate * (reward + discount * max(Q[next_state].values()))
    
    # Update V based on max Q-value
    V[state] = max(Q[state].values())


def follow_policy(state, policy, mdp):
    action = policy[state]
    next_state = max(mdp.transition_probs[state][action], key=mdp.transition_probs[state][action].get)
    return next_state



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
    grid_size = 5
    cell_size = 120
    screen_size = grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size + 200))  # Increased height further
    pygame.display.set_caption("Treasure and Trap MDP Visualization")

    num_obstacles = 3
    start, treasure, trap, obstacles = generate_random_positions(grid_size, num_obstacles)

    mdp = create_treasure_trap_mdp(grid_size, start, treasure, trap, obstacles)

    V, Q, policy, iterations = value_iteration(mdp)

    running = True
    show_policy = False
    show_q_values = False
    show_learning_process = False
    manual_learning_mode = False
    policy_demo_mode = False
    walker_pos = start

    learning_rate = 0.1
    discount = 0.9

    # Create buttons
    button_height = 30
    button_spacing = 10
    button_y1 = screen_size + 15
    button_y2 = button_y1 + button_height + button_spacing

    policy_button_rect = pygame.Rect(10, button_y1, 150, button_height)
    q_value_button_rect = pygame.Rect(170, button_y1, 150, button_height)
    learning_button_rect = pygame.Rect(330, button_y1, 150, button_height)
    manual_learning_button_rect = pygame.Rect(10, button_y2, 150, button_height)
    policy_demo_button_rect = pygame.Rect(170, button_y2, 150, button_height)
    
    button_color = LIGHT_GRAY
    policy_button_text = FONT.render("Toggle Policy", True, BLACK)
    q_value_button_text = FONT.render("Toggle Q-Values", True, BLACK)
    learning_button_text = FONT.render("Show Learning", True, BLACK)
    manual_learning_text = FONT.render("Manual Learning", True, BLACK)
    policy_demo_text = FONT.render("Policy Demo", True, BLACK)

    episode_count = 0
    max_episodes = 1000
    demo_step_delay = 500  # milliseconds

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if policy_button_rect.collidepoint(event.pos):
                    show_policy = not show_policy
                elif q_value_button_rect.collidepoint(event.pos):
                    show_q_values = not show_q_values
                elif learning_button_rect.collidepoint(event.pos):
                    show_learning_process = not show_learning_process
                    current_iteration = 0
                elif manual_learning_button_rect.collidepoint(event.pos):
                    manual_learning_mode = not manual_learning_mode
                    policy_demo_mode = False
                    if manual_learning_mode:
                        V = {s: 0 for s in mdp.states}
                        Q = {s: {a: 0 for a in mdp.actions} for s in mdp.states}
                        V[treasure] = 1.0
                        V[trap] = -1.0
                        for a in mdp.actions:
                            Q[treasure][a] = 1.0
                            Q[trap][a] = -1.0
                        walker_pos = start
                        episode_count = 0
                elif policy_demo_button_rect.collidepoint(event.pos):
                    policy_demo_mode = not policy_demo_mode
                    manual_learning_mode = False
                    if policy_demo_mode:
                        walker_pos = random.choice([s for s in mdp.states if s not in obstacles and s != treasure and s != trap])
            # ... [Keep the keyboard event handling] ...
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    show_policy = not show_policy
                elif event.key == pygame.K_q:
                    show_q_values = not show_q_values
                elif event.key == pygame.K_l:
                    show_learning_process = not show_learning_process
                    current_iteration = 0
                elif event.key == pygame.K_r:
                    start, treasure, trap, obstacles = generate_random_positions(grid_size, num_obstacles)
                    mdp = create_treasure_trap_mdp(grid_size, start, treasure, trap, obstacles)
                    V, Q, policy, iterations = value_iteration(mdp)
                    walker_pos = start
                    episode_count = 0
                elif manual_learning_mode and episode_count < max_episodes:
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        if event.key == pygame.K_UP:
                            action = "up"
                        elif event.key == pygame.K_DOWN:
                            action = "down"
                        elif event.key == pygame.K_LEFT:
                            action = "left"
                        else:
                            action = "right"
                        
                        next_state = max(mdp.transition_probs[walker_pos][action], key=mdp.transition_probs[walker_pos][action].get)
                        reward = mdp.rewards[walker_pos][action]
                        update_values(walker_pos, action, next_state, reward, V, Q, discount, learning_rate)
                        walker_pos = next_state

                        # Check if terminal state is reached
                        if walker_pos == treasure or walker_pos == trap:
                            episode_count += 1
                            walker_pos = start  # Reset to start position
                            print(f"Episode {episode_count} completed")

        screen.fill(WHITE)
        
        if show_learning_process and current_iteration < len(iterations):
            V_current, Q_current = iterations[current_iteration]
            draw_treasure_trap_hunt(screen, grid_size, cell_size, start, treasure, trap, obstacles, V_current, Q_current, policy, show_policy, show_q_values, walker_pos)
            current_iteration += 1
            if current_iteration == len(iterations):
                show_learning_process = False
            pygame.time.wait(200)  # Add a small delay between iterations
        else:
            draw_treasure_trap_hunt(screen, grid_size, cell_size, start, treasure, trap, obstacles, V, Q, policy, show_policy, show_q_values, walker_pos)

        if policy_demo_mode and walker_pos not in [treasure, trap]:
            pygame.time.wait(demo_step_delay)
            walker_pos = follow_policy(walker_pos, policy, mdp)
            if walker_pos in [treasure, trap]:
                pygame.time.wait(1000)  # Wait a second at the terminal state
                walker_pos = random.choice([s for s in mdp.states if s not in obstacles and s != treasure and s != trap])

        # Draw buttons
        pygame.draw.rect(screen, button_color, policy_button_rect)
        pygame.draw.rect(screen, button_color, q_value_button_rect)
        pygame.draw.rect(screen, button_color, learning_button_rect)
        pygame.draw.rect(screen, button_color, manual_learning_button_rect)
        pygame.draw.rect(screen, button_color, policy_demo_button_rect)
        screen.blit(policy_button_text, (policy_button_rect.x + 10, policy_button_rect.y + 5))
        screen.blit(q_value_button_text, (q_value_button_rect.x + 10, q_value_button_rect.y + 5))
        screen.blit(learning_button_text, (learning_button_rect.x + 10, learning_button_rect.y + 5))
        screen.blit(manual_learning_text, (manual_learning_button_rect.x + 10, manual_learning_button_rect.y + 5))
        screen.blit(policy_demo_text, (policy_demo_button_rect.x + 10, policy_demo_button_rect.y + 5))

        # Display text information
        text_y = button_y2 + button_height + button_spacing
        if manual_learning_mode:
            episode_text = FONT.render(f"Episode: {episode_count}", True, BLACK)
            screen.blit(episode_text, (10, text_y))
            text_y += 30

        mode_text = FONT.render(f"Mode: {'Policy Demo' if policy_demo_mode else 'Manual Learning' if manual_learning_mode else 'Observation'}", True, BLACK)
        screen.blit(mode_text, (10, text_y))
        text_y += 30

        instructions = FONT.render("SPACE/P: Policy, Q: Q-values, L: learning, R: Randomize maze", True, BLACK)
        screen.blit(instructions, (10, text_y))
        instructions = FONT.render("Arrow keys: Manual move", True, BLACK)
        screen.blit(instructions, (10, text_y + 20))



        pygame.display.flip()
        clock.tick(30)  # Increased frame rate for smoother animation

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