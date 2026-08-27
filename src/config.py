"""Configuration loading utilities.

The whole system is driven by ``configs/config.yaml``. This module loads that
file into a plain dict and exposes a couple of small helpers so that no other
module has to know where the config lives or hard-code paths.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Project root = two levels up from this file (src/config.py -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """Load the YAML configuration file into a dictionary.

    Parameters
    ----------
    path:
        Optional path to a config file. If ``None``, the default
        ``configs/config.yaml`` next to the project root is used.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    logger.debug("Loaded config from %s", cfg_path)
    return cfg


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a possibly-relative path against the project root.

    Absolute paths are returned unchanged. This keeps dataset paths in the
    config relative and portable (rule: never hard-code dataset paths).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def configure_logging(level: str = "INFO") -> None:
    """Set up root logging once, with a simple readable format."""
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
