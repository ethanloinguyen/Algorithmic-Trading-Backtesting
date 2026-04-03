"""
config_loader.py
----------------
Loads and provides access to config/config.yaml.
Singleton pattern — load once, access anywhere via get_config().

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
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "config.yaml",
)


def load_config(path: str = None) -> dict:
    """
    Load config.yaml from disk into the module-level singleton.

    Parameters
    ----------
    path : str, optional
        Explicit path to config.yaml.
        Defaults to config/config.yaml relative to project root.

    Returns
    -------
    dict — the loaded config
    """
    global _config

    if path is None:
        # Also check environment variable for container deployments
        path = os.environ.get("CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found at: {path}\n"
            f"Set CONFIG_PATH environment variable or pass path explicitly."
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