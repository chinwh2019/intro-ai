#!/usr/bin/env python3
"""
Quick start script for Reinforcement Learning Module

Usage:
  python run_rl.py                   # Run with default settings
  python run_rl.py --episodes 500    # Train for 500 episodes
  python run_rl.py --load            # Load saved model
  python run_rl.py --help            # Show help
"""

import sys
import os
import argparse

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.reinforcement_learning.main import main
from modules.reinforcement_learning.config import config, load_preset, PRESETS


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Reinforcement Learning: Snake Q-Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rl.py                        # Default settings (1000 episodes)
  python run_rl.py --preset fast_learning # Fast learning preset
  python run_rl.py --episodes 200          # Train for 200 episodes
  python run_rl.py --load                  # Load saved model
  python run_rl.py --preset visual_demo    # Slower for watching

Available presets:
  """ + "\n  ".join(f"- {name}" for name in PRESETS.keys())
    )

    parser.add_argument(
        '--preset',
        type=str,
        choices=list(PRESETS.keys()),
        help='Load a preset configuration'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        help='Number of training episodes'
    )

    parser.add_argument(
        '--load',
        action='store_true',
        help='Load existing model'
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate (alpha)'
    )

    parser.add_argument(
        '--discount',
        type=float,
        help='Discount factor (gamma)'
    )

    parser.add_argument(
        '--speed',
        type=int,
        help='Game speed'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load preset if specified
    if args.preset:
        load_preset(args.preset)

    # Override specific parameters
    if args.episodes:
        config.NUM_EPISODES = args.episodes
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.discount:
        config.DISCOUNT_FACTOR = args.discount
    if args.speed:
        config.GAME_SPEED = args.speed

    # Print configuration
    print("\nConfiguration:")
    print(f"  Episodes: {config.NUM_EPISODES}")
    print(f"  Learning Rate (α): {config.LEARNING_RATE}")
    print(f"  Discount (γ): {config.DISCOUNT_FACTOR}")
    print(f"  Epsilon: {config.EPSILON_START} → {config.EPSILON_END}")
    print(f"  Game Speed: {config.GAME_SPEED}")
    if args.load:
        print(f"  Loading model from: {config.MODEL_SAVE_PATH}")
    print()

    # Import and run directly (avoid double argument parsing)
    from modules.reinforcement_learning.main import RLTrainer

    trainer = RLTrainer(load_model=args.load)
    trainer.run()
