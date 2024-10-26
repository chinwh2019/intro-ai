import pygame
import sys
from game_environment import FlappyBirdGame
from q_learning_agent import QLearningAgent
from experiments import ExperimentManager
from utils import PathManager

def initialize_game():
    """Initialize game components with error handling"""
    try:
        game = FlappyBirdGame()
        agent = QLearningAgent()
        experiment_manager = ExperimentManager(game, agent)
        return game, agent, experiment_manager
    except Exception as e:
        print(f"Error initializing game: {e}")
        sys.exit(1)

def main():
    # Ensure data directory exists
    PathManager.get_data_dir()
    
    # Initialize game components
    game, agent, experiment_manager = initialize_game()
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    mode = "AI"
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        mode = "Player" if mode == "AI" else "AI"
                    elif event.key == pygame.K_UP and mode == "Player":
                        game.bird_velocity = game.config.FLAP_STRENGTH
                    elif event.key == pygame.K_e:
                        experiment_manager.learning_rate_experiment()
            
            if mode == "AI":
                state = game.get_state()
                action = agent.get_action(state)
                reward, next_state, done = game.step(action)
                agent.update(state, action, reward, next_state)
                
                if done:
                    game.reset()
            else:
                _, _, done = game.step(0)
                if done:
                    game.reset()
            
            game.draw(mode)
            clock.tick(60)
    
    except Exception as e:
        print(f"Error during game execution: {e}")
    
    finally:
        # Save Q-table and cleanup
        try:
            agent.save_q_table()
        except Exception as e:
            print(f"Error saving Q-table: {e}")
        
        pygame.quit()

if __name__ == "__main__":
    main()