"""
app/utils/config.py
Centralized configuration loader. Reads YAML configs and exposes typed settings.
"""

import yaml
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def load_settings(path: str = "configs/settings.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f)


def get_setting(key_path: str, default=None, config_path: str = "configs/settings.yaml"):
    """
    Dot-notation access to nested settings.
    e.g. get_setting("database.path") -> "data/autopilot.db"
    """
    settings = load_settings(config_path)
    keys = key_path.split(".")
    val = settings
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val


def setup_logging(config_path: str = "configs/settings.yaml") -> None:
    """Configure root logger from settings."""
    level_str = get_setting("logging.level", "INFO", config_path)
    fmt = get_setting(
        "logging.format",
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        config_path,
    )
    log_file = get_setting("logging.file", None, config_path)

    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level_str.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )
