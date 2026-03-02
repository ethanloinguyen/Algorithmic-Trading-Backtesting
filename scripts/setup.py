"""
scripts/setup.py
----------------
One-time setup script for new deployments.

Runs:
    1. Create all BigQuery tables
    2. Ingest Fama-French factors
    3. Ingest ticker metadata from Yahoo Finance
    4. Run universe quality filter
    5. Run historical residuals for all windows
    6. Print summary

Usage:
    python -m scripts.setup
    python -m scripts.setup --skip-historical  # Skip full historical residuals run
    python -m scripts.setup --factors-only     # Just update FF factors
"""

import argparse
import logging
import sys
import pandas as pd
from datetime import date

sys.path.insert(0, "/app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_setup(skip_historical: bool = False, factors_only: bool = False):
    from src.config_loader import load_config, get_config
    load_config()
    cfg = get_config()

    logger.info("=" * 60)
    logger.info("QUANT LEAD-LAG PIPELINE — INITIAL SETUP")
    logger.info("=" * 60)

    # ── Step 1: Create BigQuery tables ────────────────────────────────────
    logger.info("\n[1/6] Creating BigQuery tables...")
    from src.bq_schema import create_all_tables
    create_all_tables()
    logger.info("  ✓ Tables created")

    if factors_only:
        # ── Factors only mode ─────────────────────────────────────────────
        logger.info("\n[2/6] Downloading and ingesting Fama-French factors...")
        from src.residuals import ingest_ff_factors
        factor_model = cfg["residuals"]["factor_model"]
        ingest_ff_factors(factor_model)
        logger.info(f"  ✓ {factor_model} factors ingested")
        return

    # ── Step 2: Fama-French factors ───────────────────────────────────────
    logger.info("\n[2/6] Downloading and ingesting Fama-French factors...")
    from src.residuals import ingest_ff_factors
    factor_model = cfg["residuals"]["factor_model"]
    ingest_ff_factors(factor_model)
    logger.info(f"  ✓ {factor_model} factors ingested")

    # ── Step 3: Ticker metadata ───────────────────────────────────────────
    logger.info("\n[3/6] Skipping ticker metadata ingestion (using existing yfinance_stocks_data.ticker-metadata)")
    meta_df = pd.DataFrame()  # placeholder
    logger.info("  ✓ Skipped — metadata already exists in yfinance_stocks_data dataset")

    # ── Step 4: Universe quality filter ──────────────────────────────────
    logger.info("\n[4/6] Running universe quality filter...")
    from src.universe import run_universe_filter
    from datetime import datetime
    start = datetime.strptime(cfg["universe"]["start_date"], "%Y-%m-%d").date()
    end = datetime.strptime(cfg["universe"]["end_date"], "%Y-%m-%d").date()
    filter_df = run_universe_filter(start, end)
    n_valid = filter_df["is_valid"].sum()
    logger.info(f"  ✓ {n_valid} valid tickers pass quality filter")

    if n_valid < 100:
        logger.warning(
            f"  WARNING: Only {n_valid} tickers passed. "
            "Consider relaxing thresholds in config.yaml "
            "(min_market_cap, min_median_dollar_volume, min_trading_day_coverage)"
        )

    if skip_historical:
        logger.info("\n[5/6] Skipping historical residuals (--skip-historical flag set)")
        logger.info("\n[6/6] Setup complete (partial — run historical pipeline separately)")
        return

    # ── Step 5: Historical residuals ──────────────────────────────────────
    logger.info("\n[5/6] Computing historical rolling residuals (this may take a while)...")
    logger.info("  Note: This runs residuals for all windows from start_date to end_date.")
    logger.info("  For production: consider running this as a Cloud Run job.")
    from src.residuals import run_residuals_pipeline
    from src.windows import generate_rolling_windows
    windows = generate_rolling_windows()
    logger.info(f"  Processing {len(windows)} rolling windows...")
    run_residuals_pipeline(windows=windows, only_latest=False)
    logger.info("  ✓ Historical residuals computed")

    # ── Step 6: Summary ───────────────────────────────────────────────────
    logger.info("\n[6/6] Setup Summary")
    logger.info("=" * 60)
    logger.info(f"  Universe: {n_valid} valid tickers")
    logger.info(f"  Date range: {cfg['universe']['start_date']} → {cfg['universe']['end_date']}")
    logger.info(f"  Rolling windows: {len(windows)}")
    logger.info(f"  Factor model: {factor_model}")
    logger.info("")
    logger.info("  NEXT STEPS:")
    logger.info("  1. Run historical pair computation:")
    logger.info("     → Deploy pair_job.py to Cloud Run")
    logger.info("     → Trigger for all historical windows")
    logger.info("  2. Run aggregation_job.py after pair jobs complete")
    logger.info("  3. Deploy FastAPI backend")
    logger.info("  4. Deploy Cloud Scheduler for monthly trigger")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initial pipeline setup")
    parser.add_argument("--skip-historical", action="store_true",
                        help="Skip historical residuals computation")
    parser.add_argument("--factors-only", action="store_true",
                        help="Only update Fama-French factors")
    args = parser.parse_args()

    run_setup(
        skip_historical=args.skip_historical,
        factors_only=args.factors_only,
    )
