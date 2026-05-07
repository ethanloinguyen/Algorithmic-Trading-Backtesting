"""
bq_schema.py
------------
BigQuery table schema definitions and creation utilities.

All pipeline tables are defined here.
Run create_all_tables() once during initial setup.

Tables:
    market_data             — Raw OHLCV (assumed pre-existing)
    ticker_metadata         — Sector, market cap, industry
    ff_factors              — Fama-French daily factors
    filtered_universe       — Quality-filtered ticker list
    rolling_residuals       — OLS residuals per ticker per window
    pair_results_raw        — dCor + p-values per pair per lag per window
    pair_results_filtered   — FDR-corrected significant pairs
    stability_metrics       — Cross-window stability features (X_ij)
    oos_strategy_returns    — Daily OOS strategy returns per pair
    model_weights           — β regression weights + bootstrap CI
    final_network           — Final ranked pairs for API serving
    synthetic_health_log    — Monthly synthetic check results
    pipeline_run_log        — Per-run metadata and timing
"""

import logging
from typing import List

from google.cloud import bigquery
from google.cloud.exceptions import Conflict

from Algorithm.src.bq_io import get_client, get_bq_dataset
from Algorithm.src.config_loader import get_config

logger = logging.getLogger(__name__)


def _get_full_table_id(table_name: str) -> str:
    cfg = get_config()
    project = cfg["gcp"]["project_id"]
    dataset = cfg["gcp"]["bq_dataset"]
    return f"{project}.{dataset}.{table_name}"


def _create_table(
    client: bigquery.Client,
    table_id: str,
    schema: List[bigquery.SchemaField],
    partition_field: str = None,
    clustering_fields: List[str] = None,
    description: str = "",
) -> None:
    """Create a BigQuery table. Skips if already exists."""
    table = bigquery.Table(table_id, schema=schema)
    table.description = description

    if partition_field:
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )

    if clustering_fields:
        table.clustering_fields = clustering_fields

    try:
        client.create_table(table)
        logger.info(f"  ✓ Created: {table_id}")
    except Conflict:
        logger.info(f"  → Already exists: {table_id}")
    except Exception as e:
        logger.error(f"  ✗ Failed to create {table_id}: {e}")
        raise


# ── Schema Definitions ────────────────────────────────────────────────────────

def _ticker_metadata_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("name", "STRING"),
        bigquery.SchemaField("sector", "STRING"),
        bigquery.SchemaField("industry", "STRING"),
        bigquery.SchemaField("market_cap", "FLOAT64"),
        bigquery.SchemaField("exchange", "STRING"),
        bigquery.SchemaField("country", "STRING"),
        bigquery.SchemaField("updated_at", "DATE"),
    ]


def _ff_factors_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("mkt_rf", "FLOAT64"),
        bigquery.SchemaField("smb", "FLOAT64"),
        bigquery.SchemaField("hml", "FLOAT64"),
        bigquery.SchemaField("rmw", "FLOAT64"),
        bigquery.SchemaField("cma", "FLOAT64"),
        bigquery.SchemaField("rf", "FLOAT64"),
    ]


def _filtered_universe_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("is_valid", "BOOL"),
        bigquery.SchemaField("reason", "STRING"),
        bigquery.SchemaField("n_days", "INT64"),
        bigquery.SchemaField("coverage", "FLOAT64"),
        bigquery.SchemaField("median_dollar_volume", "FLOAT64"),
        bigquery.SchemaField("market_cap", "FLOAT64"),
        bigquery.SchemaField("first_date", "DATE"),
        bigquery.SchemaField("last_date", "DATE"),
        bigquery.SchemaField("filter_date", "DATE"),
    ]


def _rolling_residuals_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("run_id", "STRING", mode="NULLABLE"), # Added
        bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("window_end", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("residual", "FLOAT64"),
        bigquery.SchemaField("factor_model", "STRING"),
    ]


def _pair_results_raw_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ticker_i", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker_j", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lag", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("dcor", "FLOAT64"),
        bigquery.SchemaField("p_value", "FLOAT64"),
        bigquery.SchemaField("permutations_used", "INT64"),
        bigquery.SchemaField("sharpness", "FLOAT64"),
        bigquery.SchemaField("sharpness_entropy", "FLOAT64"),
        bigquery.SchemaField("pearson_corr", "FLOAT64"),
    ]


def _pair_results_filtered_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ticker_i", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker_j", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("lag", "INT64"),
        bigquery.SchemaField("dcor", "FLOAT64"),
        bigquery.SchemaField("q_value", "FLOAT64"),
        bigquery.SchemaField("significant", "BOOL"),
        bigquery.SchemaField("pearson_corr", "FLOAT64"),
    ]


def _stability_metrics_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("ticker_i", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker_j", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("best_lag", "INT64"),
        bigquery.SchemaField("mean_dcor", "FLOAT64"),
        bigquery.SchemaField("variance_dcor", "FLOAT64"),
        bigquery.SchemaField("frequency", "FLOAT64"),
        bigquery.SchemaField("half_life", "FLOAT64"),
        bigquery.SchemaField("half_life_r2", "FLOAT64"),
        bigquery.SchemaField("half_life_stable", "BOOL"),
        bigquery.SchemaField("sharpness", "FLOAT64"),
        bigquery.SchemaField("n_windows_observed", "INT64"),
        bigquery.SchemaField("last_updated", "DATE"),
    ]


def _oos_strategy_returns_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("ticker_i", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker_j", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("window_start", "DATE"),
        bigquery.SchemaField("lag", "INT64"),
        bigquery.SchemaField("oos_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("signal", "FLOAT64"),
        bigquery.SchemaField("position", "INT64"),
        bigquery.SchemaField("strategy_return_gross", "FLOAT64"),
        bigquery.SchemaField("strategy_return_net", "FLOAT64"),
    ]


def _model_weights_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("refit_date", "DATE"),
        bigquery.SchemaField("feature", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("weight", "FLOAT64"),
        bigquery.SchemaField("ci_lower", "FLOAT64"),
        bigquery.SchemaField("ci_upper", "FLOAT64"),
        bigquery.SchemaField("r2", "FLOAT64"),
        bigquery.SchemaField("f_statistic", "FLOAT64"),
        bigquery.SchemaField("n_pairs", "INT64"),
        bigquery.SchemaField("is_current", "BOOL"),
    ]


def _final_network_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("as_of_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ticker_i", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker_j", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("best_lag", "INT64"),
        bigquery.SchemaField("mean_dcor", "FLOAT64"),
        bigquery.SchemaField("variance_dcor", "FLOAT64"),
        bigquery.SchemaField("frequency", "FLOAT64"),
        bigquery.SchemaField("half_life", "FLOAT64"),
        bigquery.SchemaField("sharpness", "FLOAT64"),
        bigquery.SchemaField("predicted_sharpe", "FLOAT64"),
        bigquery.SchemaField("signal_strength", "FLOAT64"),
        bigquery.SchemaField("oos_sharpe_net", "FLOAT64"),
        bigquery.SchemaField("oos_dcor", "FLOAT64"),
        bigquery.SchemaField("sector_i", "STRING"),
        bigquery.SchemaField("sector_j", "STRING"),
        bigquery.SchemaField("rank", "INT64"),
        bigquery.SchemaField("centrality_i", "FLOAT64"),
        bigquery.SchemaField("centrality_j", "FLOAT64"),
    ]


def _synthetic_health_log_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("run_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("true_positive_rate", "FLOAT64"),
        bigquery.SchemaField("false_positive_rate", "FLOAT64"),
        bigquery.SchemaField("stability_rank_accuracy", "FLOAT64"),
        bigquery.SchemaField("n_planted", "INT64"),
        bigquery.SchemaField("n_null", "INT64"),
        bigquery.SchemaField("alert_tpr", "BOOL"),
        bigquery.SchemaField("alert_fpr", "BOOL"),
        bigquery.SchemaField("status", "STRING"),
    ]


def _pipeline_run_log_schema() -> List[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("run_date", "TIMESTAMP"),
        bigquery.SchemaField("window_start", "DATE"),
        bigquery.SchemaField("window_end", "DATE"),
        bigquery.SchemaField("n_pairs_processed", "INT64"),
        bigquery.SchemaField("n_significant_pairs", "INT64"),
        bigquery.SchemaField("tier3_fraction", "FLOAT64"),
        bigquery.SchemaField("cpu_hours_used", "FLOAT64"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("error_message", "STRING"),
        bigquery.SchemaField("duration_seconds", "FLOAT64"),
    ]


# ── Table Creation ────────────────────────────────────────────────────────────

def create_all_tables() -> None:
    """
    Create all pipeline tables in BigQuery.
    Safe to run multiple times — skips tables that already exist.
    Does NOT create the market_data table (assumed pre-existing).
    """
    client = get_client()
    cfg = get_config()

    logger.info("Creating BigQuery tables...")

    tables_to_create = [
        (
            cfg["tables"]["ticker_metadata"],
            _ticker_metadata_schema(),
            None, ["ticker"],
            "Ticker sector, market cap, and metadata"
        ),
        (
            cfg["tables"]["ff_factors"],
            _ff_factors_schema(),
            None, None,
            "Fama-French daily factor returns"
        ),
        (
            cfg["tables"]["filtered_universe"],
            _filtered_universe_schema(),
            None, ["ticker"],
            "Quality-filtered ticker universe"
        ),
        (
            cfg["tables"]["rolling_residuals"],
            _rolling_residuals_schema(),
            "window_start", ["ticker", "window_start"],
            "OLS residuals from FF factor regression per rolling window"
        ),
        (
            cfg["tables"]["pair_results_raw"],
            _pair_results_raw_schema(),
            "window_start", ["ticker_i", "ticker_j", "window_start"],
            "Raw dCor + permutation p-values per pair-lag-window"
        ),
        (
            cfg["tables"]["pair_results_filtered"],
            _pair_results_filtered_schema(),
            "window_start", ["ticker_i", "ticker_j", "window_start"],
            "FDR-corrected significant pairs per window"
        ),
        (
            cfg["tables"]["stability_metrics"],
            _stability_metrics_schema(),
            None, ["ticker_i", "ticker_j"],
            "Cross-window stability features (X_ij) for regression"
        ),
        (
            cfg["tables"]["oos_strategy_returns"],
            _oos_strategy_returns_schema(),
            "oos_date", ["ticker_i", "ticker_j", "oos_date"],
            "Daily OOS strategy returns per pair"
        ),
        (
            cfg["tables"]["model_weights"],
            _model_weights_schema(),
            "refit_date", ["model_version", "feature"],
            "OOS regression β weights with bootstrap CI"
        ),
        (
            cfg["tables"]["final_network"],
            _final_network_schema(),
            "as_of_date", ["ticker_i", "ticker_j", "as_of_date"],
            "Final ranked pair network for API serving"
        ),
        (
            cfg["tables"]["synthetic_health_log"],
            _synthetic_health_log_schema(),
            "run_date", None,
            "Monthly synthetic health check results"
        ),
        (
            cfg["tables"]["pipeline_run_log"],
            _pipeline_run_log_schema(),
            "run_date", ["status"],
            "Per-run pipeline metadata and timing"
        ),
    ]

    for table_key, schema, partition_field, clustering_fields, description in tables_to_create:
        table_id = _get_full_table_id(table_key)
        _create_table(
            client, table_id, schema,
            partition_field=partition_field,
            clustering_fields=clustering_fields,
            description=description,
        )

    logger.info(f"Table creation complete: {len(tables_to_create)} tables processed")