#!/usr/bin/env python3
"""
Quick start script for Search Algorithms Module

Usage:
  python run_search.py              # Run with default settings
  python run_search.py --preset fast    # Use 'fast' preset
  python run_search.py --help           # Show help
"""

import sys
import os
import argparse

# Add project root to path (parent of scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from modules.search.main import main
from modules.search.config import config, load_preset, PRESETS


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Search Algorithms Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_search.py                    # Default settings
  python run_search.py --preset fast      # Fast animation
  python run_search.py --preset simple    # Simple small maze
  python run_search.py --width 50 --height 40  # Custom maze size

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
        '--width',
        type=int,
        help='Maze width'
    )

    parser.add_argument(
        '--height',
        type=int,
        help='Maze height'
    )

    parser.add_argument(
        '--speed',
        type=float,
        help='Animation speed multiplier (e.g., 2.0 for 2x speed)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load preset if specified
    if args.preset:
        load_preset(args.preset)

    # Override specific parameters
    if args.width:
        config.MAZE_WIDTH = args.width
    if args.height:
        config.MAZE_HEIGHT = args.height
    if args.speed:
        config.ANIMATION_SPEED = args.speed

    # Print configuration
    print("\nConfiguration:")
    print(f"  Maze Size: {config.MAZE_WIDTH} x {config.MAZE_HEIGHT}")
    print(f"  Animation Speed: {config.ANIMATION_SPEED}x")
    print(f"  Cell Size: {config.CELL_SIZE}px")
    print()

    # Run main application
    main()
