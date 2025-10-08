#!/usr/bin/env python3
"""
Quick start script for MDP Module

Usage:
  python run_mdp.py                  # Run with default settings
  python run_mdp.py --preset deterministic  # Use deterministic preset
  python run_mdp.py --help               # Show help
"""

import sys
import os
import argparse

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.mdp.main import main
from modules.mdp.config import config, load_preset, PRESETS


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='MDP Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mdp.py                       # Default settings
  python run_mdp.py --preset deterministic    # No randomness
  python run_mdp.py --preset high_noise       # More stochastic
  python run_mdp.py --grid-size 8             # Larger grid

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
        '--grid-size',
        type=int,
        help='Grid size (NxN)'
    )

    parser.add_argument(
        '--noise',
        type=float,
        help='Transition noise (0.0 to 1.0)'
    )

    parser.add_argument(
        '--discount',
        type=float,
        help='Discount factor gamma (0.0 to 1.0)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load preset if specified
    if args.preset:
        load_preset(args.preset)

    # Override specific parameters
    if args.grid_size:
        config.GRID_SIZE = args.grid_size
    if args.noise is not None:
        config.NOISE = args.noise
    if args.discount is not None:
        config.DISCOUNT = args.discount

    # Print configuration
    print("\nConfiguration:")
    print(f"  Grid Size: {config.GRID_SIZE} x {config.GRID_SIZE}")
    print(f"  Discount (γ): {config.DISCOUNT}")
    print(f"  Noise: {config.NOISE}")
    print(f"  Living Reward: {config.LIVING_REWARD}")
    print()

    # Run main application
    main()
