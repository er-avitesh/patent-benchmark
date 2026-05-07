"""
Shared utilities for the patent benchmark scripts.

Every script imports config + logger from here so we have one logging
format and one place that knows how to find config.yaml.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler


# Project root = parent of `scripts/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def load_config() -> dict[str, Any]:
    """Load and return config.yaml as a dict."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """Load .env if present. Silent if not — caller checks for keys."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Rich-formatted logger. Use one per script."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    return logging.getLogger(name)


def resolve_path(relative_path: str) -> Path:
    """Convert a config-relative path to an absolute Path."""
    return PROJECT_ROOT / relative_path


def require_env_var(var_name: str) -> str:
    """Fetch an env var or raise a clear error."""
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"Environment variable {var_name} is not set. "
            f"Add it to .env (see .env.example)."
        )
    return value


# Convenience for scripts that just want a console for pretty printing
console = Console()