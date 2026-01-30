import yaml
from pathlib import Path


def load_config(config_path="configs/ttt_config.yaml"):
    """Load config from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
