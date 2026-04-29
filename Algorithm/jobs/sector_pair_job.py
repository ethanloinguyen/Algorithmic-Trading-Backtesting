"""
sector_pair_job.py
------------------
Sector-Constrained Pair Job: dCor + Permutation for In-Sector Pairs Only

Identical to pair_job.py in every respect except pair generation:
instead of all C(n, 2) combinations across the full universe, this job
generates only pairs where both tickers share the same GICS sector
(drawn from ticker_metadata.sector).

Entry point called by run_sector_local.py for each parallel partition.

Each job:
    1. Loads residuals for its window from BigQuery (shared table)
    2. Generates in-sector pairs via get_sector_pairs()
    3. Processes its assigned subset of those pairs
    4. Computes dCor + adaptive permutation at all lags
    5. Computes sharpness
    6. Batch-writes results to sector_pair_results_raw
    7. Exits
"""

import logging
import os
import sys
import uuid
from datetime import date, datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

_jobs_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_jobs_dir)
for _p in (_project_root, "/app"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Algorithm.src.bq_io import read_residuals_for_window, write_dataframe, log_pipeline_run
from Algorithm.src.config_loader import load_config, get_config
from Algorithm.src.dcor_engine import dcor_profile, compute_sharpness, get_best_lag, pearson_at_lag
from Algorithm.src.permutation import test_pair_all_lags, check_budget_guard
from Algorithm.src.universe import get_valid_tickers, get_sector_pairs, partition_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def process_pair(
    ticker_i: str,
    ticker_j: str,
    residuals_pivot: pd.DataFrame,
    lags: List[int],
    rng: np.random.Generator,
    sharpness_method: str = "entropy",
) -> List[dict]:
    """
    Process one pair: compute dCor + permutation at all lags + sharpness.

    Returns
    -------
    List of result dicts (one per lag), ready for BigQuery.
    """
    if ticker_i not in residuals_pivot.columns or ticker_j not in residuals_pivot.columns:
        return []

    pair_df = residuals_pivot[[ticker_i, ticker_j]].dropna()
    x = pair_df[ticker_i].values
    y = pair_df[ticker_j].values

    min_obs = get_config()["windows"]["min_obs_for_dcor"]
    if len(x) < min_obs or len(y) < min_obs:
        return []

    raw_dcor = dcor_profile(x, y, lags)
    sharpness = compute_sharpness(raw_dcor, method=sharpness_method)
    raw_pearson = {lag: pearson_at_lag(x, y, lag) for lag in lags}
    perm_results = test_pair_all_lags(x, y, lags, rng=rng)

    rows = []
    for lag in lags:
        pr = perm_results.get(lag, {})
        rows.append({
            "ticker_i": ticker_i,
            "ticker_j": ticker_j,
            "lag": lag,
            "dcor": pr.get("dcor"),
            "p_value": pr.get("p_value", 1.0),
            "permutations_used": pr.get("permutations_used", 0),
            "sharpness": sharpness,
            "sharpness_entropy": sharpness,
            "pearson_corr": raw_pearson.get(lag),
        })

    return rows


def run_pair_job(
    window_start: date,
    window_end: date,
    partition_id: int,
    num_partitions: int,
    run_id: str,
) -> None:
    """Main sector pair job execution."""
    cfg = get_config()
    lags = cfg["lags"]["lag_list"]
    sharpness_method = cfg["sharpness"]["method"]
    start_time = datetime.now()

    logger.info(
        f"Sector pair job starting: window={window_start}→{window_end}, "
        f"partition={partition_id}/{num_partitions}, run_id={run_id}"
    )

    # ── Load residuals (shared table with main pipeline) ──────────────────────
    resid_df = read_residuals_for_window(window_start, window_end)
    if resid_df.empty:
        logger.error(f"No residuals found for window {window_start}→{window_end}. Exiting.")
        sys.exit(1)

    resid_df["date"] = pd.to_datetime(resid_df["date"]).dt.date
    residuals_pivot = resid_df.pivot(index="date", columns="ticker", values="residual")
    logger.info(f"Loaded residuals: {residuals_pivot.shape[1]} tickers, {residuals_pivot.shape[0]} dates")

    # ── Get in-sector pairs for this partition ────────────────────────────────
    tickers = get_valid_tickers()
    all_pairs = get_sector_pairs(tickers)          # <-- only within-sector pairs
    my_pairs = partition_pairs(all_pairs, partition_id, num_partitions)

    logger.info(f"Partition {partition_id}: processing {len(my_pairs):,} in-sector pairs")

    # ── Process pairs ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=partition_id * 1000 + hash(str(window_start)) % 1000)
    all_results = []
    tier3_count = 0
    total_pairs_processed = 0
    total_rows_written = 0
    BATCH_SIZE = 500

    for ticker_i, ticker_j in my_pairs:
        rows = process_pair(
            ticker_i, ticker_j,
            residuals_pivot,
            lags, rng, sharpness_method
        )
        for row in rows:
            if row.get("permutations_used", 0) >= cfg["permutation"]["tier3_n"]:
                tier3_count += 1
        all_results.extend(rows)
        total_pairs_processed += 1

        if total_pairs_processed % 1000 == 0:
            logger.info(f"  Progress: {total_pairs_processed:,}/{len(my_pairs):,} pairs")

        if total_pairs_processed % BATCH_SIZE == 0 and all_results:
            batch_df = pd.DataFrame(all_results)
            batch_df["window_start"] = window_start
            write_dataframe(batch_df, "pair_results_raw")   # resolved to sector_pair_results_raw via config
            total_rows_written += len(batch_df)
            logger.info(f"  Flushed {len(batch_df):,} rows (total written: {total_rows_written:,})")
            all_results = []

    check_budget_guard(tier3_count, total_pairs_processed)

    if all_results:
        batch_df = pd.DataFrame(all_results)
        batch_df["window_start"] = window_start
        write_dataframe(batch_df, "pair_results_raw")
        total_rows_written += len(batch_df)
        logger.info(f"  Flushed final {len(batch_df):,} rows (total written: {total_rows_written:,})")

    if total_rows_written == 0:
        logger.warning("No results produced by this partition.")
    else:
        logger.info(f"Wrote {total_rows_written:,} total rows to sector_pair_results_raw")

    duration = (datetime.now() - start_time).total_seconds()
    log_pipeline_run({
        "run_id": f"{run_id}_p{partition_id}",
        "run_date": datetime.now(),
        "window_start": window_start,
        "window_end": window_end,
        "n_pairs_processed": total_pairs_processed,
        "n_significant_pairs": 0,
        "tier3_fraction": tier3_count / total_pairs_processed if total_pairs_processed > 0 else 0.0,
        "cpu_hours_used": 0.0,
        "status": "COMPLETE",
        "error_message": "",
        "duration_seconds": duration,
    })

    logger.info(
        f"Sector pair job complete: {total_pairs_processed:,} pairs, "
        f"{total_rows_written:,} rows written, {duration:.1f}s"
    )


if __name__ == "__main__":
    load_config()

    window_start = date.fromisoformat(os.environ["WINDOW_START"])
    window_end = date.fromisoformat(os.environ["WINDOW_END"])
    partition_id = int(os.environ["PARTITION_ID"])
    num_partitions = int(os.environ["NUM_PARTITIONS"])
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])

    run_pair_job(window_start, window_end, partition_id, num_partitions, run_id)