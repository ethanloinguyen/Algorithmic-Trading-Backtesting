"""
run_local.py
------------
Local pipeline runner — replaces Cloud Workflows + Cloud Run Jobs.

Runs the full pipeline sequentially on your local machine while
still reading from and writing results to BigQuery on GCP.

Authentication: Uses Application Default Credentials (ADC).
Run `gcloud auth application-default login` before using this script.

Usage examples:
  # Run full pipeline for a single window
  python run_local.py --window-start 2022-01-01 --window-end 2022-12-31

  # Run pipeline for all historical windows (backfill)
  python run_local.py --mode backfill

  # Run residuals only (skip pair computation)
  python run_local.py --window-start 2022-01-01 --window-end 2022-12-31 --step residuals

  # Run pair + aggregation only (residuals already computed)
  python run_local.py --window-start 2022-01-01 --window-end 2022-12-31 --step pairs

  # Run aggregation only (pairs already written to BQ)
  python run_local.py --window-start 2022-01-01 --window-end 2022-12-31 --step aggregation

  # Run just the one-time setup (create BQ tables, ingest FF factors, filter universe)
  python run_local.py --mode setup

  # Run with specific number of parallel workers (uses Python multiprocessing)
  python run_local.py --window-start 2022-01-01 --window-end 2022-12-31 --workers 4

  # Run quarterly batch (all windows within a year)
  python run_local.py --mode quarterly --year 2022
"""

import argparse
import logging
import multiprocessing
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from typing import List, Optional, Tuple

# Prevent thread oversubscription when running many worker processes.
# Each worker uses Numba JIT + NumPy (MKL/OpenBLAS); without capping these,
# each worker spawns its own thread pool, causing CPU contention on high-core VMs.
# Set before any imports that could trigger Numba or BLAS initialisation.
# These propagate to subprocesses automatically via os.environ inheritance.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Ensure project root is on path ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config, get_config
from src.windows import generate_rolling_windows, get_latest_window

# Force UTF-8 on stdout so Unicode chars (arrows, Greek letters) don't crash on Windows CP1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter(_log_fmt))

# File handler is added in main() after --log-dir is parsed so the log lands
# in the user-specified directory (important on VMs where the home partition
# may be nearly full — use --log-dir /mnt/data1/logs on paw).
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler])
logger = logging.getLogger(__name__)


# ── Step: Residuals ───────────────────────────────────────────────────────────

def run_step_residuals(window_start: date, window_end: date, run_id: str) -> bool:
    """Compute FF residuals for a single window and write to BigQuery."""
    from src.residuals import compute_residuals_for_window, run_residuals_pipeline
    from src.universe import get_valid_tickers
    from src.bq_io import write_residuals

    cfg = get_config()
    factor_model = cfg["residuals"]["factor_model"]

    logger.info(f"[RESIDUALS] Window {window_start} → {window_end}")
    try:
        tickers = get_valid_tickers()
        if not tickers:
            logger.error("No valid tickers found. Run setup first.")
            return False

        df = compute_residuals_for_window(
            window_start, window_end, tickers, factor_model, run_id=run_id
        )
        if df.empty:
            logger.warning(f"[RESIDUALS] No residuals produced for {window_start}")
            return False

        write_residuals(df, window_start)
        logger.info(f"[RESIDUALS] Done: {len(df):,} rows written")
        return True
    except Exception as e:
        logger.error(f"[RESIDUALS] Failed: {e}", exc_info=True)
        return False


# ── Step: Pairs (single partition) ───────────────────────────────────────────

def _run_partition(args: tuple) -> Tuple[int, bool, str]:
    """
    Worker function for one partition — runs in a separate process.
    Returns (partition_id, success, error_message).
    """
    window_start, window_end, partition_id, num_partitions, run_id = args

    # Re-import in subprocess (each process needs its own state)
    import sys, os, random, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Stagger startup so workers don't all finish computing at the same instant
    # and simultaneously flood BQ with writes. Cap at 10s to limit added latency.
    time.sleep(random.uniform(0, min(partition_id * 0.1, 10.0)))

    from src.config_loader import load_config
    load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml"))

    try:
        from jobs.pair_job import run_pair_job
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
) -> bool:
    """
    Run pairwise dCor + permutation for all pairs in a window.
    Uses multiprocessing to parallelize across partitions locally.

    num_workers   : number of local CPU processes to use (default 1 = sequential)
    num_partitions: number of logical partitions (default = num_workers)
    """
    if num_partitions is None:
        num_partitions = num_workers

    logger.info(
        f"[PAIRS] Window {window_start} → {window_end} | "
        f"{num_partitions} partitions across {num_workers} workers"
    )

    partition_args = [
        (window_start, window_end, pid, num_partitions, run_id)
        for pid in range(num_partitions)
    ]

    if num_workers == 1:
        # Sequential — easier to debug, no subprocess overhead
        results = []
        for args in partition_args:
            pid, success, err = _run_partition(args)
            results.append((pid, success, err))
            if not success:
                logger.error(f"  Partition {pid} FAILED: {err}")
    else:
        # Parallel across local cores
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
        logger.warning(f"[PAIRS] {len(failed)} partitions failed: {failed}")
    else:
        logger.info(f"[PAIRS] All {num_partitions} partitions complete")

    return len(failed) == 0


# ── Step: Aggregation ─────────────────────────────────────────────────────────

def run_step_aggregation(
    window_start: date,
    window_end: date,
    run_id: str,
    is_monthly_update: bool = False,
    run_synthetic: bool = True,
) -> bool:
    """Run full aggregation: FDR, OOS eval, stability, model, network, Monte Carlo, health check."""
    from jobs.aggregation_job import run_aggregation_job

    logger.info(f"[AGGREGATION] Window {window_start} → {window_end}")
    try:
        run_aggregation_job(
            window_start=window_start,
            window_end=window_end,
            run_id=run_id,
            is_monthly_update=is_monthly_update,
            run_synthetic=run_synthetic,
        )
        logger.info("[AGGREGATION] Done")
        return True
    except Exception as e:
        logger.error(f"[AGGREGATION] Failed: {e}", exc_info=True)
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
) -> bool:
    """
    Run the complete pipeline for one window.

    step options:
        "all"         — residuals → pairs → aggregation
        "residuals"   — residuals only
        "pairs"       — pairs only (residuals must already exist in BQ)
        "aggregation" — aggregation only (pairs must already exist in BQ)
        "pairs+agg"   — pairs + aggregation (skip residuals)
    """
    logger.info("=" * 70)
    logger.info(f"PIPELINE START | window={window_start}→{window_end} | run_id={run_id}")
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
        ok = run_step_pairs(window_start, window_end, run_id, num_workers, num_partitions)
        if not ok:
            logger.warning("Some pair partitions failed — continuing to aggregation anyway.")

    if step in ("all", "aggregation", "pairs+agg"):
        ok = run_step_aggregation(window_start, window_end, run_id, is_monthly_update, run_synthetic)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"PIPELINE {'COMPLETE' if ok else 'FINISHED WITH ERRORS'} | "
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
) -> None:
    """
    Run the pipeline for ALL historical windows sequentially.
    Each window runs residuals → pairs → aggregation before moving to the next.

    This is the local equivalent of submitting batches of Cloud Run jobs.
    Expect this to take hours for a full historical backfill.
    """
    windows = generate_rolling_windows(start_date, end_date)
    logger.info(f"BACKFILL: {len(windows)} windows to process (step={step})")

    failed_windows = []
    for i, (ws, we) in enumerate(windows):
        run_id = f"backfill_{ws.strftime('%Y%m%d')}_{str(uuid.uuid4())[:6]}"
        logger.info(f"\n[{i+1}/{len(windows)}] Processing window {ws} → {we}")
        success = run_full_pipeline(
            ws, we, run_id,
            step=step,
            num_workers=num_workers,
            is_monthly_update=False,
            run_synthetic=run_synthetic,
        )
        if not success:
            failed_windows.append((ws, we))
            logger.warning(f"  Window {ws}→{we} had errors — continuing to next.")

    logger.info(f"\nBACKFILL COMPLETE: {len(windows) - len(failed_windows)}/{len(windows)} succeeded")
    if failed_windows:
        logger.warning(f"Failed windows: {failed_windows}")


# ── Quarterly Batch Mode ──────────────────────────────────────────────────────

def run_quarterly_batch(year: int, step: str = "all", num_workers: int = 1, run_synthetic: bool = True) -> None:
    """
    Run the pipeline for all windows whose window_start falls in a given year.
    Equivalent to quarterly_batches.sh but runs locally.
    """
    all_windows = generate_rolling_windows()
    year_windows = [(ws, we) for ws, we in all_windows if ws.year == year]

    logger.info(f"QUARTERLY BATCH: year={year}, {len(year_windows)} windows")
    for ws, we in year_windows:
        run_id = f"quarterly_{ws.strftime('%Y%m%d')}_{str(uuid.uuid4())[:6]}"
        run_full_pipeline(ws, we, run_id, step=step, num_workers=num_workers, is_monthly_update=False, run_synthetic=run_synthetic)


# ── Setup Mode ────────────────────────────────────────────────────────────────

def run_setup(skip_historical: bool = False, factors_only: bool = False) -> None:
    """Run one-time setup: create BQ tables, ingest FF factors, filter universe."""
    from scripts.setup import run_setup as _setup
    _setup(skip_historical=skip_historical, factors_only=factors_only)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Lead-Lag Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["single", "backfill", "quarterly", "setup", "latest"],
        default="single",
        help=(
            "single     : run one window (requires --window-start/end)\n"
            "backfill   : run all historical windows\n"
            "quarterly  : run all windows in a year (requires --year)\n"
            "latest     : run the most recent window\n"
            "setup      : one-time BQ setup (tables + FF factors + universe filter)"
        ),
    )

    parser.add_argument("--window-start", type=date.fromisoformat,
                        help="Window start date (YYYY-MM-DD)")
    parser.add_argument("--window-end", type=date.fromisoformat,
                        help="Window end date (YYYY-MM-DD)")
    parser.add_argument("--year", type=int, help="Year for quarterly batch mode")

    parser.add_argument(
        "--step",
        choices=["all", "residuals", "pairs", "aggregation", "pairs+agg"],
        default="all",
        help="Which pipeline step(s) to run (default: all)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of local parallel worker processes for pair computation (default: 1). "
            "On the paw VM (96 CPUs) use --workers 90 --partitions 360."
        ),
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=None,
        help=(
            "Number of logical pair partitions (default: same as --workers). "
            "Set higher than --workers for better load balancing, e.g. 4x workers."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Directory for the timestamped pipeline log file "
            "(default: project root). On paw use --log-dir /mnt/data1/logs "
            "to avoid filling /home."
        ),
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Flag as monthly update (skips quarterly refit logic)",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help=(
            "Skip the synthetic health check (Step 11 of aggregation). "
            "Recommended for backfill runs and development iterations — "
            "saves ~2-3 min per window. Run a dedicated monthly check instead."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (auto-generated if not set)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to config.yaml (default: config/config.yaml)",
    )

    # Setup-specific flags
    parser.add_argument("--skip-historical", action="store_true",
                        help="[setup] Skip historical residuals")
    parser.add_argument("--factors-only", action="store_true",
                        help="[setup] Only update Fama-French factors")

    # Backfill date range override
    parser.add_argument("--backfill-start", type=date.fromisoformat,
                        help="[backfill] Override start date")
    parser.add_argument("--backfill-end", type=date.fromisoformat,
                        help="[backfill] Override end date")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── File log handler (deferred so --log-dir is respected) ─────────────────
    log_dir = args.log_dir if args.log_dir else PROJECT_ROOT
    os.makedirs(log_dir, exist_ok=True)
    _log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    _file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(_log_fmt))
    logging.getLogger().addHandler(_file_handler)
    logger.info(f"Logging to: {_log_file}")

    # Load config
    load_config(args.config)
    logger.info(f"Config loaded from: {args.config}")

    run_id = args.run_id or f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"

    # ── Route to correct mode ──────────────────────────────────────────────
    if args.mode == "setup":
        run_setup(
            skip_historical=args.skip_historical,
            factors_only=args.factors_only,
        )

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
        )

    elif args.mode == "backfill":
        run_backfill(
            step=args.step,
            num_workers=args.workers,
            start_date=args.backfill_start,
            end_date=args.backfill_end,
            run_synthetic=not args.skip_synthetic,
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
        )


if __name__ == "__main__":
    # Use fork on Linux/macOS: workers inherit the parent's memory space instantly,
    # avoiding repeated module imports and Numba JIT recompilation per process.
    # This is the Linux default but setting it explicitly guards against future
    # Python version changes and documents the intent.
    if sys.platform != "win32":
        multiprocessing.set_start_method("fork", force=True)
    main()
