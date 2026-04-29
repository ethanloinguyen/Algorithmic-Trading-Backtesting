"""
run_sector_local.py
-------------------
Sector-constrained pipeline runner.

Runs the exact same pipeline as run_local.py but restricts pair
computation to within-sector stock pairs only (pairs where both
tickers share the same GICS sector from ticker_metadata).

Results are written to isolated BigQuery tables prefixed with
"sector_" so this pipeline never touches the full-universe results:
    sector_pair_results_raw
    sector_pair_results_filtered
    sector_stability_metrics
    sector_oos_strategy_returns
    sector_model_weights
    sector_final_network
    sector_synthetic_health_log
    sector_pipeline_run_log

Shared inputs (rolling_residuals, market_data, ticker_metadata,
ff_factors, filtered_universe) are read from the same BQ tables as
the main pipeline — residuals do not need to be recomputed.

Authentication: Uses Application Default Credentials (ADC).
Run `gcloud auth application-default login` before using this script.

Usage examples:
  # Run full sector pipeline for a single window
  python run_sector_local.py --window-start 2022-01-01 --window-end 2022-12-31

  # Run backfill across all historical windows
  python run_sector_local.py --mode backfill

  # Pairs only (residuals already exist in BQ from main pipeline)
  python run_sector_local.py --window-start 2022-01-01 --window-end 2022-12-31 --step pairs

  # OOS step for all historical windows (parallelized, skips already-done windows)
  python run_sector_local.py --mode backfill --step oos --workers 4

  # Model refit + sector_final_network rebuild
  python run_sector_local.py --mode latest --step finalize

  # One-time setup: create sector BQ tables
  python run_sector_local.py --mode setup
"""

import argparse
import logging
import multiprocessing
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from typing import List, Optional, Set, Tuple

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from Algorithm.src.config_loader import load_config, get_config
from Algorithm.src.windows import generate_rolling_windows, get_latest_window

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter(_log_fmt))
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler])
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "sector_config.yaml")


# ── Step: Residuals ───────────────────────────────────────────────────────────
# Residuals are shared with the main pipeline — this step is provided for
# completeness but you can skip it if the main pipeline has already
# populated rolling_residuals for the window you want to analyse.

def run_step_residuals(window_start: date, window_end: date, run_id: str) -> bool:
    """Compute sector-augmented FF residuals and write to sector_rolling_residuals."""
    from Algorithm.src.sector_residuals import compute_sector_residuals_for_window
    from Algorithm.src.universe import get_valid_tickers
    from Algorithm.src.bq_io import write_residuals

    cfg = get_config()
    factor_model = cfg["residuals"]["factor_model"]

    logger.info(f"[SECTOR RESIDUALS] Window {window_start} → {window_end}")
    try:
        tickers = get_valid_tickers()
        if not tickers:
            logger.error("No valid tickers found. Run setup first.")
            return False

        df = compute_sector_residuals_for_window(
            window_start, window_end, tickers, factor_model, run_id=run_id
        )
        if df.empty:
            logger.warning(f"[SECTOR RESIDUALS] No residuals produced for {window_start}")
            return False

        write_residuals(df, window_start)
        logger.info(f"[SECTOR RESIDUALS] Done: {len(df):,} rows written")
        return True
    except Exception as e:
        logger.error(f"[SECTOR RESIDUALS] Failed: {e}", exc_info=True)
        return False


# ── Step: Sector Pairs (single partition) ────────────────────────────────────

def _run_partition(args: tuple) -> Tuple[int, bool, str]:
    """
    Worker function for one partition — runs in a separate process.
    Dispatches to sector_pair_job (in-sector pairs only).
    Returns (partition_id, success, error_message).
    """
    window_start, window_end, partition_id, num_partitions, run_id, config_path = args

    import sys, os, random, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    time.sleep(random.uniform(0, min(partition_id * 0.1, 10.0)))

    from Algorithm.src.config_loader import load_config
    load_config(config_path)

    try:
        from Algorithm.jobs.sector_pair_job import run_pair_job
        run_pair_job(window_start, window_end, partition_id, num_partitions, run_id)
        return partition_id, True, ""
    except Exception as e:
        return partition_id, False, str(e)


def run_step_pairs(
    window_start: date,
    window_end: date,
    run_id: str,
    num_workers: int = 1,
    num_partitions: int = None,
    config_path: str = DEFAULT_CONFIG,
) -> bool:
    """
    Run pairwise dCor + permutation for all in-sector pairs in a window.
    Uses multiprocessing to parallelize across partitions locally.
    """
    if num_partitions is None:
        num_partitions = num_workers

    logger.info(
        f"[SECTOR PAIRS] Window {window_start} → {window_end} | "
        f"{num_partitions} partitions across {num_workers} workers"
    )

    partition_args = [
        (window_start, window_end, pid, num_partitions, run_id, config_path)
        for pid in range(num_partitions)
    ]

    if num_workers == 1:
        results = []
        for args in partition_args:
            pid, success, err = _run_partition(args)
            results.append((pid, success, err))
            if not success:
                logger.error(f"  Partition {pid} FAILED: {err}")
    else:
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run_partition, args): args[2] for args in partition_args}
            for future in as_completed(futures):
                pid, success, err = future.result()
                results.append((pid, success, err))
                status = "✓" if success else "✗"
                logger.info(f"  Partition {pid} {status}" + (f": {err}" if not success else ""))

    failed = [pid for pid, ok, _ in results if not ok]
    if failed:
        logger.warning(f"[SECTOR PAIRS] {len(failed)} partitions failed: {failed}")
    else:
        logger.info(f"[SECTOR PAIRS] All {num_partitions} partitions complete")

    return len(failed) == 0


# ── Step: OOS ─────────────────────────────────────────────────────────────────

def _get_already_computed_oos_windows() -> Set[date]:
    """Return sector OOS windows already written (for resume support)."""
    from Algorithm.src.bq_io import get_client, full_table
    import pandas as pd
    try:
        client = get_client()
        df = client.query(
            f"SELECT DISTINCT window_start FROM `{full_table('oos_strategy_returns')}`"
        ).to_dataframe()
        done = set(pd.to_datetime(df["window_start"]).dt.date.tolist())
        logger.info(f"  {len(done)} sector windows already in oos_strategy_returns — will skip")
        return done
    except Exception:
        return set()


def _run_oos_window(args: tuple) -> Tuple[date, bool, int, str]:
    """Worker: compute OOS strategy returns for one sector window."""
    window_start, window_end, _, config_path = args

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from Algorithm.src.config_loader import load_config
    load_config(config_path)

    try:
        import pandas as pd
        from datetime import timedelta
        from Algorithm.src.bq_io import get_client, full_table, write_dataframe
        from Algorithm.src.config_loader import get_config
        from Algorithm.src.oos_model import compute_strategy_returns
        from Algorithm.src.windows import get_oos_window_for

        cfg      = get_config()["strategy"]
        lookback = cfg["zscore_lookback_days"]

        oos_start, oos_end = get_oos_window_for(window_end)

        client = get_client()
        sig_df = client.query(f"""
            SELECT ticker_i, ticker_j, lag,
                   COALESCE(pearson_corr, 1.0) AS pearson_corr
            FROM `{full_table('pair_results_filtered')}`
            WHERE window_start = '{window_start}'
              AND significant = TRUE
        """).to_dataframe()

        if sig_df.empty:
            return window_start, True, 0, "no significant sector pairs"

        all_tickers  = list(set(sig_df["ticker_i"].tolist() + sig_df["ticker_j"].tolist()))
        ticker_list  = ", ".join(f"'{t}'" for t in all_tickers)
        extended_start = oos_start - timedelta(days=lookback * 2)

        resid_df = client.query(f"""
            WITH ranked AS (
                SELECT
                    ticker,
                    DATE(date) AS date,
                    residual,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker, DATE(date)
                        ORDER BY window_start DESC
                    ) AS rn
                FROM `{full_table('rolling_residuals')}`
                WHERE DATE(date) >= '{extended_start}'
                  AND DATE(date) <= '{oos_end}'
                  AND ticker IN ({ticker_list})
            )
            SELECT ticker, date, residual
            FROM ranked
            WHERE rn = 1
            ORDER BY ticker, date
        """).to_dataframe()

        if resid_df.empty:
            return window_start, True, 0, "no residuals for OOS period"

        resid_df["date"] = pd.to_datetime(resid_df["date"]).dt.date
        resid_pivot = resid_df.pivot(index="date", columns="ticker", values="residual")

        all_records = []
        for _, row in sig_df.iterrows():
            ti, tj, lag = row["ticker_i"], row["ticker_j"], int(row["lag"])
            direction = float(row.get("pearson_corr", 1.0) or 1.0)
            if ti not in resid_pivot.columns or tj not in resid_pivot.columns:
                continue

            oos_returns = compute_strategy_returns(
                resid_pivot[ti].dropna(),
                resid_pivot[tj].dropna(),
                lag, oos_start, oos_end,
                direction=direction,
            )
            if oos_returns.empty:
                continue

            oos_returns["ticker_i"]     = ti
            oos_returns["ticker_j"]     = tj
            oos_returns["lag"]          = lag
            oos_returns["window_start"] = window_start
            all_records.append(oos_returns)

        if not all_records:
            return window_start, True, 0, "strategy returned no records"

        result = pd.concat(all_records, ignore_index=True)
        write_dataframe(result, "oos_strategy_returns", write_disposition="WRITE_APPEND")
        return window_start, True, len(result), ""

    except Exception as e:
        return window_start, False, 0, str(e)


def run_step_oos(
    windows: List[Tuple[date, date]],
    num_workers: int = 1,
    skip_done: bool = True,
    config_path: str = DEFAULT_CONFIG,
) -> bool:
    """Compute sector OOS strategy returns for a list of windows."""
    if skip_done:
        already_done = _get_already_computed_oos_windows()
        to_compute = [(ws, we) for ws, we in windows if ws not in already_done]
    else:
        to_compute = windows

    if not to_compute:
        logger.info("[SECTOR OOS] All windows already computed — nothing to do")
        return True

    logger.info(
        f"[SECTOR OOS] {len(to_compute)} windows to compute "
        f"({len(windows) - len(to_compute)} already done) | "
        f"{num_workers} workers"
    )

    run_id = f"sector_oos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args_list = [(ws, we, run_id, config_path) for ws, we in to_compute]

    total_records = 0
    failed = []

    if num_workers == 1:
        for i, args in enumerate(args_list, 1):
            ws, success, n, msg = _run_oos_window(args)
            total_records += n
            status = "✓" if success else "✗"
            logger.info(
                f"  [{i}/{len(args_list)}] {ws} {status} "
                f"({n:,} records)" + (f" — {msg}" if msg else "")
            )
            if not success:
                failed.append(ws)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run_oos_window, args): args[0] for args in args_list}
            completed = 0
            for future in as_completed(futures):
                ws, success, n, msg = future.result()
                completed += 1
                total_records += n
                status = "✓" if success else "✗"
                logger.info(
                    f"  [{completed}/{len(args_list)}] {ws} {status} "
                    f"({n:,} records)" + (f" — {msg}" if msg else "")
                )
                if not success:
                    failed.append(ws)

    logger.info(
        f"[SECTOR OOS] Complete: {total_records:,} total records | "
        f"{len(failed)} windows failed"
    )
    if failed:
        logger.warning(f"  Failed windows: {failed}")

    return len(failed) == 0


# ── Step: Finalize ────────────────────────────────────────────────────────────

def run_step_finalize(dry_run: bool = False) -> bool:
    """Model refit + sector_final_network rebuild (single global pass)."""
    import pandas as pd
    from Algorithm.src.bq_io import (
        get_client, full_table, write_dataframe,
        read_oos_strategy_returns,
        upsert_final_network,
    )
    from Algorithm.src.bootstrap import run_model_refit, compute_predicted_sharpe, compute_signal_strength
    from Algorithm.src.network import build_directed_graph, compute_centrality
    from Algorithm.src.oos_model import compute_sharpe, compute_global_oos_dcor
    from Algorithm.src.stability import compute_stability_metrics

    cfg = get_config()
    logger.info("[SECTOR FINALIZE] Starting model refit + sector_final_network rebuild")

    logger.info("[SECTOR FINALIZE] Step 1: Computing global OOS Sharpe per sector pair...")
    all_returns = read_oos_strategy_returns()
    if all_returns.empty:
        logger.error("[SECTOR FINALIZE] sector_oos_strategy_returns is empty — run OOS step first")
        return False

    results = []
    for (ti, tj), group in all_returns.groupby(["ticker_i", "ticker_j"]):
        group     = group.sort_values("oos_date")
        net_ret   = group["strategy_return_net"].values
        gross_ret = group["strategy_return_gross"].values
        if len(net_ret) < 30:
            continue
        results.append({
            "ticker_i":         ti,
            "ticker_j":         tj,
            "oos_sharpe_net":   compute_sharpe(net_ret),
            "oos_sharpe_gross": compute_sharpe(gross_ret),
            "n_oos_days":       len(net_ret),
            "best_lag":         int(group["lag"].mode().iloc[0]),
        })

    global_sharpe_df = pd.DataFrame(results)
    logger.info(f"  Global OOS Sharpe for {len(global_sharpe_df):,} sector pairs")

    logger.info("[SECTOR FINALIZE] Step 2: Recomputing stability metrics...")
    stability_df = compute_stability_metrics()
    if stability_df.empty:
        logger.error("[SECTOR FINALIZE] sector_stability_metrics is empty")
        return False

    logger.info("[SECTOR FINALIZE] Step 3: Fitting model weights...")
    if dry_run:
        feature_weights = {f: 0.1 for f in ["mean_dcor","variance_dcor","frequency","half_life","sharpness"]}
        feature_weights["intercept"] = 0.0
    else:
        _, feature_weights = run_model_refit(stability_df, global_sharpe_df)
    logger.info(f"  Weights: {feature_weights}")

    logger.info("[SECTOR FINALIZE] Step 4: Computing predicted_sharpe and signal_strength...")
    predicted_sharpe = compute_predicted_sharpe(stability_df, feature_weights)
    signal_strength  = compute_signal_strength(
        predicted_sharpe,
        lo_pct=cfg["oos"]["sharpe_winsorize_pct"][0],
        hi_pct=cfg["oos"]["sharpe_winsorize_pct"][1],
    )
    stability_df = stability_df.copy()
    stability_df["predicted_sharpe"] = predicted_sharpe.values
    stability_df["signal_strength"]  = signal_strength.values

    logger.info("[SECTOR FINALIZE] Step 4b: Computing global OOS dCor...")
    oos_dcor_df = compute_global_oos_dcor(global_sharpe_df)

    final_df = stability_df.merge(
        global_sharpe_df[["ticker_i", "ticker_j", "oos_sharpe_net"]],
        on=["ticker_i", "ticker_j"],
        how="left",
    )
    if not oos_dcor_df.empty:
        final_df = final_df.merge(
            oos_dcor_df[["ticker_i", "ticker_j", "oos_dcor"]],
            on=["ticker_i", "ticker_j"],
            how="left",
        )

    logger.info("[SECTOR FINALIZE] Step 5: Adding sector info...")
    client  = get_client()
    meta_df = client.query(
        f"SELECT ticker, sector FROM `{full_table('ticker_metadata')}`"
    ).to_dataframe()
    sector_map = dict(zip(meta_df["ticker"], meta_df["sector"]))

    final_df["sector_i"]   = final_df["ticker_i"].map(sector_map).fillna("Unknown")
    final_df["sector_j"]   = final_df["ticker_j"].map(sector_map).fillna("Unknown")
    final_df["as_of_date"] = date.today()
    if "oos_dcor" not in final_df.columns:
        final_df["oos_dcor"] = None
    final_df["rank"] = (
        final_df["signal_strength"].rank(ascending=False, method="first").astype(int)
    )

    logger.info("[SECTOR FINALIZE] Step 6: Computing network centrality...")
    G = build_directed_graph(final_df)
    centrality_df = compute_centrality(G)
    if not centrality_df.empty:
        cent_map = dict(zip(centrality_df["ticker"], centrality_df["eigenvector_centrality"]))
        final_df["centrality_i"] = final_df["ticker_i"].map(cent_map).fillna(0.0)
        final_df["centrality_j"] = final_df["ticker_j"].map(cent_map).fillna(0.0)
    else:
        final_df["centrality_i"] = 0.0
        final_df["centrality_j"] = 0.0

    network_cols = [
        "as_of_date","ticker_i","ticker_j","best_lag",
        "mean_dcor","variance_dcor","frequency","half_life",
        "sharpness","predicted_sharpe","signal_strength",
        "oos_sharpe_net","oos_dcor","sector_i","sector_j",
        "rank","centrality_i","centrality_j",
    ]
    available  = [c for c in network_cols if c in final_df.columns]
    network_df = final_df[available].copy()

    logger.info(f"[SECTOR FINALIZE] Writing sector_final_network: {len(network_df):,} pairs")
    if not dry_run:
        upsert_final_network(network_df)
        logger.info("[SECTOR FINALIZE] Done — sector_final_network updated in BigQuery")
    else:
        logger.info("[SECTOR FINALIZE] [DRY RUN] Would write sector_final_network — skipping")

    return True


# ── Step: Aggregation ─────────────────────────────────────────────────────────

def run_step_aggregation(
    window_start: date,
    window_end: date,
    run_id: str,
    is_monthly_update: bool = False,
    run_synthetic: bool = True,
) -> bool:
    """Run full aggregation against sector tables (FDR, OOS eval, stability, model, network)."""
    from Algorithm.jobs.aggregation_job import run_aggregation_job

    logger.info(f"[SECTOR AGGREGATION] Window {window_start} → {window_end}")
    try:
        run_aggregation_job(
            window_start=window_start,
            window_end=window_end,
            run_id=run_id,
            is_monthly_update=is_monthly_update,
            run_synthetic=run_synthetic,
        )
        logger.info("[SECTOR AGGREGATION] Done")
        return True
    except Exception as e:
        logger.error(f"[SECTOR AGGREGATION] Failed: {e}", exc_info=True)
        return False


# ── Full Single-Window Pipeline ───────────────────────────────────────────────

def run_full_pipeline(
    window_start: date,
    window_end: date,
    run_id: str,
    step: str = "all",
    num_workers: int = 1,
    num_partitions: int = None,
    is_monthly_update: bool = False,
    run_synthetic: bool = True,
    config_path: str = DEFAULT_CONFIG,
) -> bool:
    logger.info("=" * 70)
    logger.info(f"SECTOR PIPELINE START | window={window_start}→{window_end} | run_id={run_id}")
    logger.info(f"  step={step} | workers={num_workers} | monthly={is_monthly_update}")
    logger.info("=" * 70)

    start_time = datetime.now()
    ok = True

    if step in ("all", "residuals"):
        ok = run_step_residuals(window_start, window_end, run_id)
        if not ok and step == "all":
            logger.error("Residuals failed — aborting pipeline.")
            return False

    if step in ("all", "pairs", "pairs+agg"):
        ok = run_step_pairs(
            window_start, window_end, run_id,
            num_workers, num_partitions, config_path
        )
        if not ok:
            logger.warning("Some pair partitions failed — continuing to aggregation anyway.")

    if step in ("all", "aggregation", "pairs+agg"):
        ok = run_step_aggregation(window_start, window_end, run_id, is_monthly_update, run_synthetic)

    if step == "oos":
        ok = run_step_oos(
            [(window_start, window_end)], num_workers=1,
            skip_done=True, config_path=config_path
        )

    if step == "finalize":
        ok = run_step_finalize()

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"SECTOR PIPELINE {'COMPLETE' if ok else 'FINISHED WITH ERRORS'} | "
        f"{elapsed:.1f}s | window={window_start}→{window_end}"
    )
    return ok


# ── Backfill Mode ─────────────────────────────────────────────────────────────

def run_backfill(
    step: str = "all",
    num_workers: int = 1,
    start_date: date = None,
    end_date: date = None,
    run_synthetic: bool = True,
    config_path: str = DEFAULT_CONFIG,
) -> None:
    windows = generate_rolling_windows(start_date, end_date)
    logger.info(f"SECTOR BACKFILL: {len(windows)} windows (step={step})")

    if step == "oos":
        run_step_oos(windows, num_workers=num_workers, skip_done=True, config_path=config_path)
        return

    if step == "finalize":
        run_step_finalize()
        return

    failed_windows = []
    for i, (ws, we) in enumerate(windows):
        run_id = f"sector_backfill_{ws.strftime('%Y%m%d')}_{str(uuid.uuid4())[:6]}"
        logger.info(f"\n[{i+1}/{len(windows)}] Processing window {ws} → {we}")
        success = run_full_pipeline(
            ws, we, run_id,
            step=step,
            num_workers=num_workers,
            is_monthly_update=False,
            run_synthetic=run_synthetic,
            config_path=config_path,
        )
        if not success:
            failed_windows.append((ws, we))
            logger.warning(f"  Window {ws}→{we} had errors — continuing.")

    logger.info(
        f"\nSECTOR BACKFILL COMPLETE: "
        f"{len(windows) - len(failed_windows)}/{len(windows)} succeeded"
    )
    if failed_windows:
        logger.warning(f"Failed windows: {failed_windows}")


# ── Quarterly Batch Mode ──────────────────────────────────────────────────────

def run_quarterly_batch(
    year: int,
    step: str = "all",
    num_workers: int = 1,
    run_synthetic: bool = True,
    config_path: str = DEFAULT_CONFIG,
) -> None:
    all_windows  = generate_rolling_windows()
    year_windows = [(ws, we) for ws, we in all_windows if ws.year == year]
    logger.info(f"SECTOR QUARTERLY BATCH: year={year}, {len(year_windows)} windows")

    if step == "oos":
        run_step_oos(year_windows, num_workers=num_workers, skip_done=True, config_path=config_path)
        return

    for ws, we in year_windows:
        run_id = f"sector_quarterly_{ws.strftime('%Y%m%d')}_{str(uuid.uuid4())[:6]}"
        run_full_pipeline(
            ws, we, run_id,
            step=step, num_workers=num_workers,
            is_monthly_update=False, run_synthetic=run_synthetic,
            config_path=config_path,
        )


# ── Setup Mode ────────────────────────────────────────────────────────────────

def run_setup() -> None:
    """Create sector-specific BQ tables (sector_pair_results_raw, etc.)."""
    from Algorithm.src.bq_schema import create_all_tables
    logger.info("[SECTOR SETUP] Creating sector BigQuery tables...")
    create_all_tables()
    logger.info("[SECTOR SETUP] Done")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sector Lead-Lag Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["single", "backfill", "quarterly", "setup", "latest"],
        default="single",
    )
    parser.add_argument("--window-start", type=date.fromisoformat)
    parser.add_argument("--window-end",   type=date.fromisoformat)
    parser.add_argument("--year",         type=int)
    parser.add_argument(
        "--step",
        choices=["all", "residuals", "pairs", "aggregation", "pairs+agg", "oos", "finalize"],
        default="all",
    )
    parser.add_argument("--workers",    type=int, default=1)
    parser.add_argument("--partitions", type=int, default=None)
    parser.add_argument("--log-dir",    type=str, default=None)
    parser.add_argument("--monthly",    action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--run-id",     type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to sector config (default: config/sector_config.yaml)",
    )
    parser.add_argument("--backfill-start", type=date.fromisoformat)
    parser.add_argument("--backfill-end",   type=date.fromisoformat)

    return parser.parse_args()


def main():
    args = parse_args()

    log_dir = args.log_dir if args.log_dir else PROJECT_ROOT
    os.makedirs(log_dir, exist_ok=True)
    _log_file = os.path.join(log_dir, f"sector_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    _file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(_log_fmt))
    logging.getLogger().addHandler(_file_handler)
    logger.info(f"Logging to: {_log_file}")

    load_config(args.config)
    logger.info(f"Sector config loaded from: {args.config}")

    run_id = args.run_id or f"sector_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"

    if args.mode == "setup":
        run_setup()

    elif args.mode == "single":
        if not args.window_start or not args.window_end:
            logger.error("--window-start and --window-end are required for single mode")
            sys.exit(1)
        run_full_pipeline(
            window_start=args.window_start,
            window_end=args.window_end,
            run_id=run_id,
            step=args.step,
            num_workers=args.workers,
            num_partitions=args.partitions,
            is_monthly_update=args.monthly,
            run_synthetic=not args.skip_synthetic,
            config_path=args.config,
        )

    elif args.mode == "latest":
        ws, we = get_latest_window()
        logger.info(f"Latest window: {ws} → {we}")
        run_full_pipeline(
            window_start=ws,
            window_end=we,
            run_id=run_id,
            step=args.step,
            num_workers=args.workers,
            num_partitions=args.partitions,
            is_monthly_update=True,
            run_synthetic=not args.skip_synthetic,
            config_path=args.config,
        )

    elif args.mode == "backfill":
        run_backfill(
            step=args.step,
            num_workers=args.workers,
            start_date=args.backfill_start,
            end_date=args.backfill_end,
            run_synthetic=not args.skip_synthetic,
            config_path=args.config,
        )

    elif args.mode == "quarterly":
        if not args.year:
            logger.error("--year is required for quarterly mode")
            sys.exit(1)
        run_quarterly_batch(
            year=args.year,
            step=args.step,
            num_workers=args.workers,
            run_synthetic=not args.skip_synthetic,
            config_path=args.config,
        )


if __name__ == "__main__":
    if sys.platform != "win32":
        multiprocessing.set_start_method("fork", force=True)
    main()