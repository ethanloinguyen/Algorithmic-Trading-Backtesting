"""
windows.py
----------
Rolling window generation utilities.

Generates (window_start, window_end) tuples for the full
historical date range defined in config.yaml.

Window logic:
    - window_days : 252 (1 trading year)
    - step_days   : 63  (quarterly step)

OOS window logic:
    - OOS window immediately follows the training window
    - OOS length = step_days (63 trading days ≈ 1 quarter)
    - This ensures no lookahead bias
"""

import logging
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from Algorithm.src.config_loader import get_config

logger = logging.getLogger(__name__)


def generate_rolling_windows(
    start_date: date = None,
    end_date: date = None,
    window_days: int = None,
    step_days: int = None,
) -> List[Tuple[date, date]]:
    """
    Generate (window_start, window_end) tuples snapped to strict 
    calendar quarters (Jan 1, Apr 1, Jul 1, Oct 1) for rolling analysis.

    Windows are calendar-day based. The window contains all trading days
    between window_start and window_end inclusive.

    Parameters
    ----------
    start_date : first window_start date (defaults to config universe.start_date)
    end_date   : last possible window_end date (defaults to config universe.end_date)
    window_days : window length in calendar days (defaults to config windows.window_days)
    step_days   : step between windows in calendar days (defaults to config windows.step_days)

    Returns
    -------
    List of (window_start, window_end) date tuples, ordered chronologically.
    """
    cfg = get_config()

    if start_date is None:
        start_date = date.fromisoformat(cfg["universe"]["start_date"])
    if end_date is None:
        # Crucial: Use Dec 31, 2025 as your "Data Ceiling"
        end_date = date.fromisoformat(cfg["universe"]["end_date"])

    # Create a range of Quarter Starts using pandas (very robust)
    # 'QS' means Quarter Start (Jan 1, Apr 1, etc.)
    quarter_starts = pd.date_range(start=start_date, end=end_date, freq='QS').date

    windows = []
    for q_start in quarter_starts:
        # window_start is the first day of the quarter
        # window_end is exactly 1 year minus 1 day later (to keep it 365 days)
        # Or, if you want 1 full calendar year:
        win_start = q_start
        win_end = (pd.Timestamp(q_start) + pd.offsets.DateOffset(years=1) - pd.offsets.Day(1)).date()
        
        if win_end > end_date:
            break
            
        windows.append((win_start, win_end))

    logger.info(f"Generated {len(windows)} strict quarterly windows up to {end_date}")
    return windows


def get_latest_window() -> Tuple[date, date]:
    """
    Return the most recent complete rolling window.
    Used in monthly update mode to process only the newest window.

    Returns
    -------
    (window_start, window_end) for the latest window.
    """
    windows = generate_rolling_windows()
    if not windows:
        raise ValueError("No windows generated. Check config dates.")
    return windows[-1]


def get_oos_window_for(window_end: date) -> Tuple[date, date]:
    """
    Return the OOS evaluation window that immediately follows
    a given training window end date.

    OOS window = [window_end + 1 day, window_end + step_days]

    Parameters
    ----------
    window_end : end date of the training window

    Returns
    -------
    (oos_start, oos_end) tuple
    """
    cfg = get_config()
    step_days = cfg["windows"]["step_days"]

    oos_start = window_end + timedelta(days=1)
    oos_end = window_end + timedelta(days=step_days)
    return oos_start, oos_end


def get_oos_windows(
    start_date: date = None,
    end_date: date = None,
) -> List[Tuple[date, date, date, date]]:
    """
    Return all (window_start, window_end, oos_start, oos_end) tuples.

    Useful for iterating over training + OOS window pairs together.

    Returns
    -------
    List of 4-tuples: (train_start, train_end, oos_start, oos_end)
    """
    train_windows = generate_rolling_windows(start_date, end_date)
    result = []
    for ws, we in train_windows:
        oos_start, oos_end = get_oos_window_for(we)
        result.append((ws, we, oos_start, oos_end))
    return result


def window_to_label(window_start: date, window_end: date) -> str:
    """Return a human-readable label for a window."""
    return f"{window_start.strftime('%Y-%m')}→{window_end.strftime('%Y-%m')}"


def date_to_nearest_window_start(
    target_date: date,
    start_date: date = None,
    step_days: int = None,
) -> Optional[date]:
    """
    Find the window_start that is closest to (and not after) target_date.

    Useful for looking up which window covers a given date.

    Returns
    -------
    date or None if target_date is before first window start.
    """
    cfg = get_config()
    if start_date is None:
        start_date = date.fromisoformat(cfg["universe"]["start_date"])
    if step_days is None:
        step_days = cfg["windows"]["step_days"]

    if target_date < start_date:
        return None

    delta_days = (target_date - start_date).days
    n_steps = delta_days // step_days
    nearest_start = start_date + timedelta(days=n_steps * step_days)
    return nearest_start