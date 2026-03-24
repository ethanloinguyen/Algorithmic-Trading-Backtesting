"""
aggregation_job.py
------------------
Cloud Run Job: Aggregation + Model Update

Runs on a SINGLE machine after all pair partition jobs complete.
Orchestrated by Cloud Workflows.

Steps:
    1.  Pull pair_results_raw for current window
    2.  Apply Benjamini-Hochberg FDR → pair_results_filtered
    3.  Compute OOS strategy returns for new window
    4.  Append to oos_strategy_returns
    5.  Recompute global OOS Sharpe per pair
    6.  Update stability metrics across all windows
    7.  Check if quarterly refit is due → refit or use frozen β
    8.  Compute predicted_sharpe + signal_strength for all pairs
    9.  Update final_network
    10. Recompute graph centrality
    11. Run Monte Carlo for top pairs
    12. Run synthetic health check
    13. Log pipeline run summary
"""

import logging
import os
import sys
import uuid
from datetime import date, datetime

import numpy as np
import pandas as pd

_agg_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_agg_dir)
for _p in (_proj_root, "/app"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.bq_io import (
    get_client, full_table, write_dataframe,
    read_pair_results_raw, read_stability_metrics,
    read_model_weights, read_oos_strategy_returns,
    upsert_final_network, log_pipeline_run
)
from src.bootstrap import (
    run_model_refit, compute_predicted_sharpe,
    compute_signal_strength, FEATURES
)
from src.config_loader import load_config, get_config
from src.fdr import apply_fdr_pipeline
from src.monte_carlo import run_monte_carlo_pipeline
from src.network import run_network_pipeline, build_directed_graph, compute_centrality
from src.oos_model import (
    run_oos_evaluation_for_window, compute_global_oos_sharpe
)
from src.stability import compute_stability_metrics
from src.synthetic import run_synthetic_health_check
from src.windows import get_latest_window, get_oos_window_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def is_quarterly_refit_due() -> bool:
    """Check if today is a quarterly model refit month (Jan, Apr, Jul, Oct)."""
    cfg = get_config()["model"]
    return date.today().month in cfg["refit_months"]


def get_frozen_weights() -> dict:
    """Load current frozen β weights from BigQuery."""
    weights_df = read_model_weights()
    if weights_df.empty:
        logger.warning("No model weights found. Cannot compute stability scores.")
        return {}
    weights = dict(zip(weights_df["feature"], weights_df["weight"]))
    logger.info(f"Loaded frozen weights: {weights}")
    return weights


def run_aggregation_job(
    window_start: date,
    window_end: date,
    run_id: str,
    is_monthly_update: bool = True,
    run_synthetic: bool = True,
) -> None:
    """
    Full aggregation pipeline.

    Parameters
    ----------
    window_start, window_end : the new window just computed
    run_id : unique identifier for this pipeline run
    is_monthly_update : if True, skip historical recomputation
    run_synthetic : if False, skip the synthetic health check (Step 11).
                    Overridden to False when synthetic.enabled=false in config.
    """
    cfg = get_config()
    start_time = datetime.now()
    n_significant = 0
    status = "COMPLETE"
    error_msg = ""

    try:
        logger.info(f"=== AGGREGATION JOB START === run_id={run_id}, window={window_start}→{window_end}")

        # ── Step 1: Pull raw pair results for this window ─────────────────
        logger.info("Step 1: Loading raw pair results...")
        raw_df = read_pair_results_raw(window_start)
        logger.info(f"  {len(raw_df):,} raw pair-lag records loaded")

        # ── Step 2: Apply FDR ─────────────────────────────────────────────
        logger.info("Step 2: Applying Benjamini-Hochberg FDR...")
        filtered_df = apply_fdr_pipeline(window_start, raw_df)
        n_significant = int(filtered_df["significant"].sum()) if not filtered_df.empty else 0
        logger.info(f"  {n_significant:,} pairs significant after FDR")

        # Significant pairs for OOS evaluation
        if filtered_df.empty:
            sig_pairs = pd.DataFrame(columns=["ticker_i", "ticker_j", "lag"])
        else:
            sig_pairs = filtered_df[filtered_df["significant"]][["ticker_i", "ticker_j", "lag"]]

        # ── Step 3: Compute OOS strategy returns for this window ──────────
        logger.info("Step 3: Computing OOS strategy returns...")
        oos_start, oos_end = get_oos_window_for(window_end)
        oos_returns_df = run_oos_evaluation_for_window(
            window_start, window_end, oos_start, oos_end, sig_pairs
        )

        if not oos_returns_df.empty:
            write_dataframe(oos_returns_df, "oos_strategy_returns", write_disposition="WRITE_APPEND")
            logger.info(f"  {len(oos_returns_df):,} OOS return records written")
        else:
            logger.warning("  No OOS strategy returns computed (no significant pairs or no data)")

        # ── Step 4: Recompute global OOS Sharpe per pair ──────────────────
        logger.info("Step 4: Computing global OOS Sharpe per pair...")
        global_sharpe_df = compute_global_oos_sharpe()
        logger.info(f"  Global Sharpe computed for {len(global_sharpe_df):,} pairs")

        # ── Step 5: Recompute stability metrics ───────────────────────────
        logger.info("Step 5: Computing stability metrics across all windows...")
        stability_df = compute_stability_metrics()
        logger.info(f"  Stability metrics for {len(stability_df):,} pairs")

        # ── Step 6: Model weights (refit or frozen) ───────────────────────
        if is_quarterly_refit_due() and not is_monthly_update:
            # Quarterly refit
            logger.info("Step 6: Quarterly model refit...")
            if not stability_df.empty and not global_sharpe_df.empty:
                _, feature_weights = run_model_refit(stability_df, global_sharpe_df)
            else:
                logger.warning("Insufficient data for refit. Using frozen weights.")
                feature_weights = get_frozen_weights()
        else:
            logger.info("Step 6: Using frozen β weights (monthly update mode)...")
            feature_weights = get_frozen_weights()
            if not feature_weights:
                # First run — need to fit model
                logger.info("  No frozen weights found. Running initial model fit...")
                if not stability_df.empty and not global_sharpe_df.empty:
                    _, feature_weights = run_model_refit(stability_df, global_sharpe_df)

        # ── Step 7: Compute predicted Sharpe + Signal Strength ────────────
        logger.info("Step 7: Computing predicted Sharpe and Signal Strength...")
        if feature_weights and not stability_df.empty:
            predicted_sharpe = compute_predicted_sharpe(stability_df, feature_weights)
            signal_strength = compute_signal_strength(
                predicted_sharpe,
                lo_pct=cfg["oos"]["sharpe_winsorize_pct"][0],
                hi_pct=cfg["oos"]["sharpe_winsorize_pct"][1],
            )
            stability_df["predicted_sharpe"] = predicted_sharpe.values
            stability_df["signal_strength"] = signal_strength.values
        else:
            stability_df["predicted_sharpe"] = 0.0
            stability_df["signal_strength"] = 50.0

        # ── Step 8: Build final_network ───────────────────────────────────
        logger.info("Step 8: Building final_network...")
        as_of_date = date.today()

        # Merge stability metrics with OOS Sharpe and ticker metadata
        client = get_client()
        metadata_query = f"""
            SELECT ticker, sector, industry
            FROM `{full_table('ticker_metadata')}`
        """
        metadata_df = client.query(metadata_query).to_dataframe()
        sector_map = dict(zip(metadata_df["ticker"], metadata_df["sector"]))

        # Merge OOS Sharpe into stability
        if not global_sharpe_df.empty and not stability_df.empty:
            final_df = stability_df.merge(
                global_sharpe_df[["ticker_i", "ticker_j", "oos_sharpe_net"]],
                on=["ticker_i", "ticker_j"],
                how="left"
            )
        elif not stability_df.empty:
            final_df = stability_df.copy()
            final_df["oos_sharpe_net"] = None
        else:
            logger.warning("Step 8: No stability data to build final_network. Skipping.")
            final_df = pd.DataFrame()

        # Add sector info
        if not final_df.empty:
            final_df["sector_i"] = final_df["ticker_i"].map(sector_map).fillna("Unknown")
            final_df["sector_j"] = final_df["ticker_j"].map(sector_map).fillna("Unknown")
            final_df["as_of_date"] = as_of_date

            # Add OOS dCor as secondary metric (placeholder)
            final_df["oos_dcor"] = None

            # Rank by signal strength
            if "signal_strength" in final_df.columns:
                final_df["rank"] = final_df["signal_strength"].rank(ascending=False, method="first").astype(int)
            else:
                final_df["rank"] = 0

            # Select columns matching final_network schema
            network_cols = [
                "as_of_date", "ticker_i", "ticker_j", "best_lag",
                "mean_dcor", "variance_dcor", "frequency", "half_life",
                "sharpness", "predicted_sharpe", "signal_strength",
                "oos_sharpe_net", "oos_dcor", "sector_i", "sector_j", "rank"
            ]
            network_cols_available = [c for c in network_cols if c in final_df.columns]
            network_df = final_df[network_cols_available].copy()

            upsert_final_network(network_df)
            logger.info(f"  final_network updated: {len(network_df):,} pairs")
        else:
            logger.warning("  Skipping final_network update — no data.")
            network_df = pd.DataFrame()

        # ── Step 9: Recompute network centrality ──────────────────────────
        logger.info("Step 9: Computing network centrality...")
        centrality_df, network_json = run_network_pipeline(as_of_date)

        # Inject centrality scores back into final_network
        if not centrality_df.empty and not network_df.empty:
            centrality_i = dict(zip(centrality_df["ticker"], centrality_df["eigenvector_centrality"]))
            network_df["centrality_i"] = network_df["ticker_i"].map(centrality_i).fillna(0.0)
            network_df["centrality_j"] = network_df["ticker_j"].map(centrality_i).fillna(0.0)
            upsert_final_network(network_df)

        # ── Step 10: Monte Carlo for top pairs ────────────────────────────
        logger.info("Step 10: Running Monte Carlo simulation...")
        all_oos_returns = read_oos_strategy_returns()
        if not all_oos_returns.empty:
            mc_results = run_monte_carlo_pipeline(all_oos_returns, as_of_date, top_n_pairs=200)
            logger.info(f"  Monte Carlo complete for top pairs")

        # ── Step 11: Synthetic health check ───────────────────────────────
        synthetic_enabled = cfg.get("synthetic", {}).get("enabled", True)
        if run_synthetic and synthetic_enabled:
            logger.info("Step 11: Running synthetic health check...")
            health_result = run_synthetic_health_check()
            logger.info(f"  Health check: {health_result['status']}")
        else:
            logger.info("Step 11: Synthetic health check skipped.")

    except Exception as e:
        logger.error(f"Aggregation job failed: {e}", exc_info=True)
        status = "ERROR"
        error_msg = str(e)
        raise

    finally:
        # ── Step 12: Log run ──────────────────────────────────────────────
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run({
            "run_id": run_id,
            "run_date": datetime.now(),
            "window_start": window_start,
            "window_end": window_end,
            "n_pairs_processed": 0,
            "n_significant_pairs": n_significant,
            "tier3_fraction": 0.0,
            "cpu_hours_used": 0.0,
            "status": status,
            "error_message": error_msg,
            "duration_seconds": duration,
        })
        logger.info(f"=== AGGREGATION JOB {'COMPLETE' if status == 'COMPLETE' else 'FAILED'} === {duration:.1f}s")


if __name__ == "__main__":
    load_config()

    window_start = date.fromisoformat(os.environ["WINDOW_START"])
    window_end = date.fromisoformat(os.environ["WINDOW_END"])
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])
    is_monthly = os.environ.get("IS_MONTHLY_UPDATE", "true").lower() == "true"

    run_aggregation_job(window_start, window_end, run_id, is_monthly_update=is_monthly)
