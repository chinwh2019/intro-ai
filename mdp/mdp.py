import random
import time

class MDP:
    def __init__(self, states, actions, transition_probs, rewards, discount):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.discount = discount

def value_iteration(mdp, epsilon=0.001):
    """
    Perform value iteration algorithm on the given MDP.
    
    Args:
    mdp (MDP): The Markov Decision Process
    epsilon (float): Convergence threshold
    
    Returns:
    dict: Optimal value function
    dict: Optimal policy
    """
    V = {s: 0 for s in mdp.states}
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            V[s] = max(sum(mdp.transition_probs[s][a][s1] * (mdp.rewards[s][a] + mdp.discount * V[s1])
                           for s1 in mdp.states)
                       for a in mdp.actions)
            delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    
    policy = {}
    for s in mdp.states:
        policy[s] = max(mdp.actions,
                        key=lambda a: sum(mdp.transition_probs[s][a][s1] * (mdp.rewards[s][a] + mdp.discount * V[s1])
                                          for s1 in mdp.states))
    return V, policy

def print_grid_world(grid_size, start, goal, policy=None):
    """Print a visual representation of the grid world."""
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == start:
                print("S", end=" ")
            elif (i, j) == goal:
                print("G", end=" ")
            elif policy and (i, j) in policy:
                print(policy[(i, j)][0].upper(), end=" ")
            else:
                print(".", end=" ")
        print()

def create_grid_world_mdp(grid_size, start, goal):
    """Create an MDP for a simple grid world."""
    states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    actions = ["up", "down", "left", "right"]
    
    transition_probs = {s: {a: {s1: 0.0 for s1 in states} for a in actions} for s in states}
    rewards = {s: {a: -1 for a in actions} for s in states}
    
    # Set transition probabilities and rewards
    for i in range(grid_size):
        for j in range(grid_size):
            s = (i, j)
            for a in actions:
                if s == goal:
                    transition_probs[s][a][s] = 1.0
                    rewards[s][a] = 0
                else:
                    if a == "up":
                        s1 = (max(i-1, 0), j)
                    elif a == "down":
                        s1 = (min(i+1, grid_size-1), j)
                    elif a == "left":
                        s1 = (i, max(j-1, 0))
                    elif a == "right":
                        s1 = (i, min(j+1, grid_size-1))
                    transition_probs[s][a][s1] = 1.0
                    rewards[s][a] = -1 if s1 != goal else 10
    
    return MDP(states, actions, transition_probs, rewards, discount=0.9)

def main():
    print("Welcome to the Markov Decision Process (MDP) and Value Iteration Algorithm Explainer!")
    print("This application will guide you through the concepts using a simple grid world example.")
    
    # Get user input for grid world size
    while True:
        try:
            grid_size = int(input("\nEnter the size of the grid world (e.g., 3 for a 3x3 grid): "))
            if grid_size < 2 or grid_size > 5:
                raise ValueError
            break
        except ValueError:
            print("Please enter a valid integer between 2 and 5.")

    # Set start and goal states
    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)

    print("\nHere's your grid world:")
    print("S: Start, G: Goal, .: Empty cell")
    print_grid_world(grid_size, start, goal)

    # Create MDP
    mdp = create_grid_world_mdp(grid_size, start, goal)

    print("\nNow, let's apply the Value Iteration algorithm to find the optimal policy.")
    print("The algorithm will iterate until the values converge.")

    # Run Value Iteration
    V, policy = value_iteration(mdp)

    print("\nValue Iteration complete! Here's the optimal policy:")
    print_grid_world(grid_size, start, goal, policy)

    print("\nLet's break down what happened:")
    print("1. We initialized a value of 0 for each state.")
    print("2. In each iteration, we updated the value of each state based on the best action.")
    print("3. We continued this process until the values converged (changed very little).")
    print("4. Finally, we determined the optimal policy based on these converged values.")

    print("\nHere are the final state values:")
    for i in range(grid_size):
        for j in range(grid_size):
            print(f"({i},{j}): {V[(i,j)]:.2f}", end="  ")
        print()

    print("\nKey Concepts:")
    print("- States: Each cell in the grid")
    print("- Actions: Up, Down, Left, Right")
    print("- Rewards: -1 for each move, 10 for reaching the goal")
    print("- Discount factor: 0.9 (future rewards are slightly less valuable)")
    print("- Policy: The best action to take in each state")

    print("\nThank you for using the MDP and Value Iteration Explainer!")

if __name__ == "__main__":
    main()