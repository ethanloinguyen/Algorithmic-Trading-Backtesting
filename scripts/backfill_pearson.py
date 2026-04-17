"""
scripts/backfill_pearson.py
---------------------------
Standalone script to compute and backfill pearson_corr for all significant
pairs already stored in BigQuery, without re-running the full pipeline.

Why this exists:
    dCor detects *that* a lead-lag relationship exists but is direction-blind.
    pearson_corr at the same lag captures the sign (+/-) needed to trade
    correctly (long vs. short the follower stock). This script backfills that
    value for pairs the algorithm has already identified as significant.

Reads from:
    - pair_results_filtered  (identifies which pairs/windows/lags to process)
    - rolling_residuals      (provides the same training-window data dCor used)

Writes to (via DML UPDATE on staging table — no rows added or removed):
    - pair_results_raw       (pearson_corr at all 5 lags for significant pairs)
    - pair_results_filtered  (pearson_corr at best_lag for significant pairs)

Safe to re-run: by default skips pairs where pearson_corr is already populated.
Use --force to recompute and overwrite existing values.

Usage:
    python -m scripts.backfill_pearson
    python -m scripts.backfill_pearson --dry-run
    python -m scripts.backfill_pearson --window-start 2022-01-01
    python -m scripts.backfill_pearson --force
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime

import numpy as np
import pandas as pd

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_scripts_dir)
for _p in (_proj_root, "/app"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_significant_pairs(client, cfg: dict, window_start_filter: date = None, force: bool = False) -> pd.DataFrame:
    """
    Load significant pairs from pair_results_filtered.

    If force=False, only returns rows where pearson_corr IS NULL (not yet computed).
    If force=True, returns all significant pairs regardless.

    Returns DataFrame with: window_start, ticker_i, ticker_j, lag (best_lag)
    """
    from src.bq_io import full_table

    window_clause = f"AND window_start = '{window_start_filter}'" if window_start_filter else ""
    pearson_clause = "" if force else "AND (pearson_corr IS NULL)"

    query = f"""
        SELECT window_start, ticker_i, ticker_j, lag
        FROM `{full_table('pair_results_filtered')}`
        WHERE significant = TRUE
          {window_clause}
          {pearson_clause}
        ORDER BY window_start, ticker_i, ticker_j
    """
    df = client.query(query).to_dataframe()
    df["window_start"] = pd.to_datetime(df["window_start"]).dt.date
    logger.info(f"Found {len(df):,} significant pair-window rows needing pearson_corr")
    return df


def _load_residuals_for_window(client, window_start: date, tickers: list) -> pd.DataFrame:
    """
    Load training-window residuals for a set of tickers.
    Uses the same window_start filter as the original dCor computation.
    """
    from src.bq_io import full_table

    ticker_list = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT ticker, DATE(date) AS date, residual
        FROM `{full_table('rolling_residuals')}`
        WHERE window_start = '{window_start}'
          AND ticker IN ({ticker_list})
        ORDER BY ticker, date
    """
    df = client.query(query).to_dataframe()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _compute_pearson_for_window(
    sig_pairs: pd.DataFrame,
    pivot: pd.DataFrame,
    lags: list,
) -> pd.DataFrame:
    """
    For all significant pairs in one window, compute pearson_corr at every lag.

    Returns DataFrame with: window_start, ticker_i, ticker_j, lag, pearson_corr
    Suitable for updating both pair_results_raw (all lags) and
    pair_results_filtered (best_lag only, subset of this result).
    """
    from src.dcor_engine import pearson_at_lag

    records = []
    window_start = sig_pairs["window_start"].iloc[0]

    for _, row in sig_pairs.iterrows():
        ti, tj = row["ticker_i"], row["ticker_j"]

        if ti not in pivot.columns or tj not in pivot.columns:
            logger.debug(f"  Skipping {ti}/{tj}: not in residuals pivot")
            continue

        pair_df = pivot[[ti, tj]].dropna()
        if len(pair_df) < 20:
            logger.debug(f"  Skipping {ti}/{tj}: insufficient data ({len(pair_df)} rows)")
            continue

        x = pair_df[ti].values
        y = pair_df[tj].values

        for lag in lags:
            pearson = pearson_at_lag(x, y, lag)
            records.append({
                "window_start": window_start,
                "ticker_i": ti,
                "ticker_j": tj,
                "lag": lag,
                "pearson_corr": pearson,
            })

    return pd.DataFrame(records)


def _process_window_worker(args: tuple):
    """
    Top-level worker for ProcessPoolExecutor.
    Creates its own BigQuery client (not picklable across processes).

    Args:
        args: (window_start, window_pairs_records, lags)

    Returns:
        list of pearson record dicts, or None if nothing computed
    """
    window_start, window_pairs_records, lags = args

    from src.bq_io import get_client
    from src.config_loader import load_config

    load_config()
    client = get_client()

    window_pairs = pd.DataFrame(window_pairs_records)
    window_pairs["window_start"] = pd.to_datetime(window_pairs["window_start"]).dt.date

    tickers = list(set(
        window_pairs["ticker_i"].tolist() + window_pairs["ticker_j"].tolist()
    ))

    resid_df = _load_residuals_for_window(client, window_start, tickers)
    if resid_df.empty:
        return None

    pivot = resid_df.pivot(index="date", columns="ticker", values="residual")
    window_pearson = _compute_pearson_for_window(window_pairs, pivot, lags)

    if window_pearson.empty:
        return None

    return window_pearson.to_dict("records")


def _write_staging_table(client, cfg: dict, df: pd.DataFrame, staging_table_name: str) -> str:
    """
    Write the pearson updates to a temporary staging table in BigQuery.
    Returns the fully qualified staging table ID.
    """
    from google.cloud import bigquery

    project = cfg["gcp"]["project_id"]
    dataset = cfg["gcp"]["bq_dataset"]
    table_id = f"{project}.{dataset}.{staging_table_name}"

    # Drop any existing staging table from a previous interrupted run
    try:
        client.delete_table(table_id)
        logger.info(f"Dropped existing staging table: {table_id}")
    except Exception:
        pass

    schema = [
        bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ticker_i", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker_j", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lag", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("pearson_corr", "FLOAT64"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=schema,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logger.info(f"Wrote {len(df):,} rows to staging table {table_id}")
    return table_id


def _run_dml_update(client, target_table_id: str, staging_table_id: str, label: str) -> int:
    """
    Run a DML UPDATE on target_table using pearson_corr values from staging_table.
    Matches on (window_start, ticker_i, ticker_j, lag).
    Returns number of rows updated.
    """
    dml = f"""
        UPDATE `{target_table_id}` t
        SET t.pearson_corr = s.pearson_corr
        FROM `{staging_table_id}` s
        WHERE t.window_start = s.window_start
          AND t.ticker_i = s.ticker_i
          AND t.ticker_j = s.ticker_j
          AND t.lag = s.lag
    """
    logger.info(f"Running DML UPDATE on {label}...")
    job = client.query(dml)
    job.result()
    n_updated = job.num_dml_affected_rows or 0
    logger.info(f"  {label}: {n_updated:,} rows updated")
    return n_updated


def _delete_staging_table(client, staging_table_id: str) -> None:
    try:
        client.delete_table(staging_table_id)
        logger.info(f"Deleted staging table: {staging_table_id}")
    except Exception as e:
        logger.warning(f"Could not delete staging table {staging_table_id}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backfill(
    window_start_filter: date = None,
    dry_run: bool = False,
    force: bool = False,
    workers: int = 1,
) -> None:
    from src.bq_io import get_client, full_table
    from src.config_loader import load_config, get_config

    load_config()
    cfg = get_config()
    client = get_client()
    lags = cfg["lags"]["lag_list"]

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("PEARSON BACKFILL — START")
    if dry_run:
        logger.info("  DRY RUN: no writes will be made")
    if force:
        logger.info("  FORCE: recomputing all significant pairs (even already-populated)")
    logger.info("=" * 60)

    # ── Step 1: Load which pairs need pearson_corr ────────────────────────
    sig_df = _load_significant_pairs(client, cfg, window_start_filter, force)

    if sig_df.empty:
        logger.info("Nothing to compute — all significant pairs already have pearson_corr.")
        return

    all_windows = sorted(sig_df["window_start"].unique())
    logger.info(f"Processing {len(all_windows)} windows, {len(sig_df):,} total pair-window rows")

    # ── Step 2: Compute pearson per window ────────────────────────────────
    logger.info(f"Processing {len(all_windows)} windows | workers={workers}")
    all_pearson_rows = []

    worker_args = [
        (
            window_start,
            sig_df[sig_df["window_start"] == window_start].to_dict("records"),
            lags,
        )
        for window_start in all_windows
    ]

    if workers == 1:
        for i, args in enumerate(worker_args, 1):
            window_start = args[0]
            logger.info(f"[{i}/{len(all_windows)}] Window {window_start}")
            records = _process_window_worker(args)
            if records is None:
                logger.warning(f"  No pearson values computed for window {window_start} — skipping")
                continue
            df = pd.DataFrame(records)
            n_pos = (df["pearson_corr"] > 0).sum()
            n_neg = (df["pearson_corr"] < 0).sum()
            logger.info(f"  Computed {len(df)} pearson values ({n_pos} positive, {n_neg} negative)")
            all_pearson_rows.append(df)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for args in worker_args:
                fut = executor.submit(_process_window_worker, args)
                futures[fut] = args[0]  # window_start

            completed = 0
            for fut in as_completed(futures):
                window_start = futures[fut]
                completed += 1
                try:
                    records = fut.result()
                except Exception as e:
                    logger.warning(f"  [{completed}/{len(all_windows)}] Window {window_start} failed: {e}")
                    continue
                if records is None:
                    logger.warning(f"  [{completed}/{len(all_windows)}] Window {window_start} — no values computed, skipping")
                    continue
                df = pd.DataFrame(records)
                n_pos = (df["pearson_corr"] > 0).sum()
                n_neg = (df["pearson_corr"] < 0).sum()
                logger.info(
                    f"  [{completed}/{len(all_windows)}] Window {window_start}: "
                    f"{len(df)} pearson values ({n_pos} positive, {n_neg} negative)"
                )
                all_pearson_rows.append(df)

    if not all_pearson_rows:
        logger.warning("No pearson values were computed. Check residuals data.")
        return

    full_pearson_df = pd.concat(all_pearson_rows, ignore_index=True)
    logger.info(f"Total pearson values computed: {len(full_pearson_df):,} rows")

    if dry_run:
        logger.info("DRY RUN — sample of computed values:")
        sample = full_pearson_df.dropna(subset=["pearson_corr"]).head(20)
        logger.info(f"\n{sample.to_string(index=False)}")
        logger.info("\nDirection breakdown (at best_lag per pair):")
        best_lag_rows = full_pearson_df.merge(
            sig_df[["window_start", "ticker_i", "ticker_j", "lag"]],
            on=["window_start", "ticker_i", "ticker_j", "lag"],
            how="inner"
        )
        if not best_lag_rows.empty:
            pos = (best_lag_rows["pearson_corr"] > 0).sum()
            neg = (best_lag_rows["pearson_corr"] < 0).sum()
            null = best_lag_rows["pearson_corr"].isna().sum()
            logger.info(f"  Positive (i up → j up):   {pos:,}")
            logger.info(f"  Negative (i up → j down): {neg:,}")
            logger.info(f"  Null (insufficient data): {null:,}")
        logger.info("DRY RUN complete — no writes made.")
        return

    # ── Step 3: Write staging table ───────────────────────────────────────
    staging_name = f"_pearson_backfill_staging_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    staging_table_id = _write_staging_table(client, cfg, full_pearson_df, staging_name)

    try:
        # ── Step 4: Update pair_results_raw (all lags) ────────────────────
        raw_table_id = full_table("pair_results_raw")
        n_raw = _run_dml_update(client, raw_table_id, staging_table_id, "pair_results_raw")

        # ── Step 5: Update pair_results_filtered (best_lag only) ──────────
        # The staging table has all lags; the DML JOIN on (window_start, ticker_i,
        # ticker_j, lag) means only the best_lag row in filtered will match.
        filtered_table_id = full_table("pair_results_filtered")
        n_filtered = _run_dml_update(client, filtered_table_id, staging_table_id, "pair_results_filtered")

    finally:
        # ── Step 6: Clean up staging table ───────────────────────────────
        _delete_staging_table(client, staging_table_id)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("PEARSON BACKFILL — COMPLETE")
    logger.info(f"  pair_results_raw updated:      {n_raw:,} rows")
    logger.info(f"  pair_results_filtered updated: {n_filtered:,} rows")
    logger.info(f"  Duration: {duration:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill pearson_corr for significant pairs in BigQuery"
    )
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Restrict to a single window (YYYY-MM-DD). Omit to process all windows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and preview values without writing to BigQuery.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite pearson_corr even for pairs that already have it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    args = parser.parse_args()

    window_filter = date.fromisoformat(args.window_start) if args.window_start else None

    run_backfill(
        window_start_filter=window_filter,
        dry_run=args.dry_run,
        force=args.force,
        workers=args.workers,
    )
