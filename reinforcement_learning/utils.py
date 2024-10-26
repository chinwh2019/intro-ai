import os
import json
from pathlib import Path


class PathManager:
    @staticmethod
    def get_project_root():
        """Returns project root folder."""
        return Path(__file__).parent.parent

    @staticmethod
    def get_data_dir():
        """Returns data directory path and creates it if it doesn't exist."""
        data_dir = PathManager.get_project_root() / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @staticmethod
    def get_q_table_path():
        """Returns the path to the Q-table file."""
        return PathManager.get_data_dir() / 'q_table.json'


class DataManager:
    @staticmethod
    def save_q_table(q_table):
        """Saves Q-table to JSON file."""
        q_table_path = PathManager.get_q_table_path()
        with open(q_table_path, 'w') as f:
            json.dump({str(k): v for k, v in q_table.items()}, f)

    @staticmethod
    def load_q_table():
        """Loads Q-table from JSON file."""
        q_table_path = PathManager.get_q_table_path()
        if q_table_path.exists():
            with open(q_table_path, 'r') as f:
                return json.load(f)
        return {}