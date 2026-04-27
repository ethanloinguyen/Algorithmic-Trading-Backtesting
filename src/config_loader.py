"""
config_loader.py
----------------
Loads and provides access to config/config.yaml.
Singleton pattern — load once, access anywhere via get_config().

Supports both local (config/config.yaml next to project root) and
Cloud Run (/app/config/config.yaml) environments automatically.

Usage:
    from src.config_loader import load_config, get_config

    load_config()               # Call once at startup
    cfg = get_config()          # Call anywhere after that
    cfg["fdr"]["alpha"]         # -> 0.05
"""

import os
from typing import Optional

import yaml

_config: Optional[dict] = None


def _find_default_config() -> str:
    """
    Search common locations for config.yaml in priority order:
    1. CONFIG_PATH env var
    2. /app/config/config.yaml (Cloud Run container)
    3. <project_root>/config/config.yaml (local: one level up from src/)
    4. ./config/config.yaml (cwd fallback)
    """
    # 1. Explicit env var always wins
    env_path = os.environ.get("CONFIG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Cloud Run standard location
    if os.path.exists("/app/config/config.yaml"):
        return "/app/config/config.yaml"

    # 3. Local: src/ → project root (one level up from this file's directory)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    local_path = os.path.join(project_root, "config", "config.yaml")
    if os.path.exists(local_path):
        return local_path

    # 4. CWD fallback
    cwd_path = os.path.join(os.getcwd(), "config", "config.yaml")
    if os.path.exists(cwd_path):
        return cwd_path

    return local_path  # Return best guess; FileNotFoundError will explain


def load_config(path: str = None) -> dict:
    """
    Load config.yaml from disk into the module-level singleton.

    Parameters
    ----------
    path : str, optional
        Explicit path to config.yaml. Auto-detected if not provided.

    Returns
    -------
    dict — the loaded config
    """
    global _config

    if path is None:
        path = _find_default_config()

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found at: {path}\n"
            f"Options:\n"
            f"  1. Set CONFIG_PATH environment variable\n"
            f"  2. Place config.yaml at {path}\n"
            f"  3. Pass path explicitly: load_config('/path/to/config.yaml')"
        )

    with open(path, "r") as f:
        _config = yaml.safe_load(f)

    return _config


def get_config() -> dict:
    """
    Return the loaded config dict.
    Raises RuntimeError if load_config() has not been called yet.
    """
    global _config
    if _config is None:
        # Attempt auto-load from default path as convenience
        try:
            load_config()
        except FileNotFoundError:
            raise RuntimeError(
                "Config not loaded. Call load_config() before get_config(), "
                "or set the CONFIG_PATH environment variable."
            )
    return _config


def get_gcp_project() -> str:
    """Convenience: return GCP project ID from config."""
    return get_config()["gcp"]["project_id"]


def get_bq_dataset() -> str:
    """Convenience: return BigQuery dataset name from config."""
    return get_config()["gcp"]["bq_dataset"]


def get_table_name(key: str) -> str:
    """
    Return the BigQuery table name for a given config key.

    Parameters
    ----------
    key : str — key in config.yaml tables section
        e.g. "pair_results_raw", "final_network", "model_weights"

    Returns
    -------
    str — table name as configured
    """
    tables = get_config().get("tables", {})
    if key not in tables:
        raise KeyError(f"Table key '{key}' not found in config tables section.")
    return tables[key]