"""
This module houses helpful tools and functions to run the Aerial-Illuminate.
"""

import yaml


def load_config(file_path):
    """Load configuration file."""
    with open(file_path, 'r') as conf:
        configs = yaml.load(conf, Loader=yaml.FullLoader)
        return configs
