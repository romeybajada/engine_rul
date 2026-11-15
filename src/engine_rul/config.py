import os
from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def load_config(path=None):
    path = path or os.environ.get("CONFIG_PATH", "config/config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def data_dir(config) -> Path:
    """The data folder. DATA_DIR in the environment overrides config.yaml, for every command."""
    return Path(os.environ.get("DATA_DIR", config["data_dir"]))
