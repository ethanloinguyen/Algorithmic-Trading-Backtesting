"""
bq_io.py
--------
All BigQuery read/write operations for the pipeline.

Single source of truth for:
    - Client management (singleton)
    - Table name resolution
    - DataFrame writes with schema enforcement
    - All domain-specific read functions

Every other module imports from here. No raw BigQuery calls
should exist outside this file.
"""

import logging
import random
import time
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import ResourceExhausted

from src.config_loader import get_config, get_gcp_project, get_bq_dataset

logger = logging.getLogger(__name__)

# ── Client Singleton ──────────────────────────────────────────────────────────

_client: Optional[bigquery.Client] = None


def get_client() -> bigquery.Client:
    """
    Return a singleton BigQuery client.
    Initializes on first call using application default credentials.
    In Cloud Run, credentials come from the attached service account automatically.
    """
    global _client
    if _client is None:
        cfg = get_config()
        project = cfg["gcp"]["project_id"]
        _client = bigquery.Client(project=project)
        logger.info(f"BigQuery client initialized for project: {project}")
    return _client


# ── Table Name Utilities ──────────────────────────────────────────────────────

def get_bq_dataset() -> str:
    """Return the configured BigQuery dataset name."""
    return get_config()["gcp"]["bq_dataset"]


# Source tables read from the original dataset, never written by this pipeline.
_SOURCE_TABLES = {"market_data", "ticker_metadata"}


def full_table(table_key: str) -> str:
    """
    Return fully qualified BigQuery table ID: project.dataset.table

    Source tables (market_data, ticker_metadata) are always resolved against
    gcp.source_dataset so that changing gcp.bq_dataset for local isolation
    never breaks reads of the underlying price/metadata data.

    All pipeline output tables resolve against gcp.bq_dataset.
    """
    cfg = get_config()
    project = cfg["gcp"]["project_id"]
    default_dataset = cfg["gcp"]["bq_dataset"]
    source_dataset = cfg["gcp"].get("source_dataset", default_dataset)

    # Get the value from the 'tables' section of config.yaml
    table_path = cfg["tables"].get(table_key, table_key)

    # If already fully qualified (contains a dot), just prepend project
    if "." in table_path:
        return f"{project}.{table_path}"

    # Source tables always come from source_dataset
    if table_key in _SOURCE_TABLES:
        return f"{project}.{source_dataset}.{table_path}"

    # All pipeline output tables go to bq_dataset
    return f"{project}.{default_dataset}.{table_path}"


# ── Generic Write ─────────────────────────────────────────────────────────────

# Maximum attempts and base delay for 429 rate-limit retries.
# Base waits: 5s, 10s, 20s, 40s, 80s (exponential backoff + random jitter).
# Jitter spreads retrying workers so they don't all retry in lock-step
# when many workers hit the rate limit simultaneously (common with 90 workers).
_WRITE_MAX_ATTEMPTS = 6
_WRITE_BASE_DELAY_S = 5


def write_dataframe(
    df: pd.DataFrame,
    table_key: str,
    write_disposition: str = "WRITE_APPEND",
    schema: List[bigquery.SchemaField] = None,
) -> None:
    """
    Write a pandas DataFrame to BigQuery with exponential backoff retry.

    Retries up to _WRITE_MAX_ATTEMPTS times on 429 ResourceExhausted errors,
    which occur when too many workers write to the same partitioned table
    simultaneously (BigQuery limit: ~50 partition updates per 10 seconds).

    Parameters
    ----------
    df : DataFrame to write
    table_key : config tables key (e.g. "pair_results_raw") OR literal table name
    write_disposition : "WRITE_APPEND" | "WRITE_TRUNCATE" | "WRITE_EMPTY"
    schema : optional explicit schema; if None, BigQuery auto-detects
    """
    if df is None or df.empty:
        logger.warning(f"write_dataframe called with empty DataFrame for table '{table_key}'. Skipping.")
        return

    client = get_client()
    table_id = full_table(table_key)

    # Convert date/datetime columns to avoid type issues
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=(schema is None),
        schema=schema if schema else [],
    )

    delay = _WRITE_BASE_DELAY_S
    for attempt in range(1, _WRITE_MAX_ATTEMPTS + 1):
        try:
            job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
            job.result()
            logger.info(f"Wrote {len(df):,} rows to {table_id} ({write_disposition})")
            return
        except ResourceExhausted as e:
            if attempt == _WRITE_MAX_ATTEMPTS:
                logger.error(f"Failed to write to {table_id}: {e}")
                raise
            jitter = random.uniform(0, delay * 0.5)
            logger.warning(
                f"BQ rate limit hit writing to {table_id} "
                f"(attempt {attempt}/{_WRITE_MAX_ATTEMPTS}), retrying in {delay + jitter:.1f}s..."
            )
            time.sleep(delay + jitter)
            delay *= 2
        except Exception as e:
            logger.error(f"Failed to write to {table_id}: {e}")
            raise


# ── Residuals ─────────────────────────────────────────────────────────────────
# 3/1 took out idempotency, can put back 
def write_residuals(df: pd.DataFrame, window_start: date) -> None:
    """
    Write rolling residuals for a window.
    Ensures table exists with correct schema before writing.
    """
    client = get_client()
    table_id = full_table("rolling_residuals")

    # 1. Define the Schema (Crucial for the pair_job to work)
    schema = [
        bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("window_end", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("residual", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("factor_model", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("run_id", "STRING", mode="NULLABLE"), # Added run_id
    ]

    # 2. Force Table Creation if it doesn't exist
    try:
        client.get_table(table_id)
    except NotFound:
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        logger.info(f"Created table {table_id}")

    # 3. If we actually have data, delete old window and write new data
    if df is not None and not df.empty:
        delete_query = f"DELETE FROM `{table_id}` WHERE window_start = '{window_start}'"
        try:
            client.query(delete_query).result()
        except Exception as e:
            logger.warning(f"Could not delete existing residuals: {e}")

        write_dataframe(df, "rolling_residuals", write_disposition="WRITE_APPEND", schema=schema)
    else:
        logger.error(
            f"!!! CRITICAL: No residuals to write for window starting {window_start}. "
            f"This usually means the inner join between market_data and ff_factors "
            f"resulted in < 60 days of data for all tickers."
        )


def read_residuals_for_window(
    window_start: date,
    window_end: date,
) -> pd.DataFrame:
    """
    Read all residuals for a specific training window.

    Returns
    -------
    DataFrame with: window_start, window_end, ticker, date, residual, factor_model
    """
    client = get_client()
    query = f"""
        SELECT window_start, window_end, ticker, DATE(date) AS date, residual, factor_model
        FROM `{full_table('rolling_residuals')}`
        WHERE window_start = '{window_start}'
        AND window_end = '{window_end}'
        ORDER BY ticker, date
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} residual rows for window {window_start}→{window_end}")
    return df


# ── Pair Results ──────────────────────────────────────────────────────────────

def write_pair_results_raw(df: pd.DataFrame) -> None:
    """Append raw pair results (dCor + p-values) for one partition."""
    write_dataframe(df, "pair_results_raw", write_disposition="WRITE_APPEND")


def read_pair_results_raw(window_start: date) -> pd.DataFrame:
    """
    Read all pair results (all lags) for a given window.

    Returns
    -------
    DataFrame with: ticker_i, ticker_j, lag, dcor, p_value,
                    permutations_used, sharpness
    """
    client = get_client()
    query = f"""
        SELECT ticker_i, ticker_j, lag, dcor, p_value,
               permutations_used, sharpness
        FROM `{full_table('pair_results_raw')}`
        WHERE window_start = '{window_start}'
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} raw pair-lag records for window {window_start}")
    return df


def read_all_pair_results_filtered() -> pd.DataFrame:
    """
    Read all FDR-filtered significant pair results across ALL windows.
    Used by stability.py to compute cross-window metrics.

    Returns
    -------
    DataFrame with: window_start, ticker_i, ticker_j, lag, dcor, q_value, significant
    """
    client = get_client()
    query = f"""
        SELECT window_start, ticker_i, ticker_j, lag, dcor, q_value, significant
        FROM `{full_table('pair_results_filtered')}`
        WHERE significant = TRUE
        ORDER BY window_start, ticker_i, ticker_j
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} filtered pair records across all windows")
    return df


# ── Daily Prices ──────────────────────────────────────────────────────────────

def read_daily_prices(
    tickers: List[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Read OHLCV market data for given tickers and date range.
    Computes log_return from close prices.

    Returns
    -------
    DataFrame with: ticker, date, open, high, low, close, volume, log_return
    """
    client = get_client()

    if not tickers:
        return pd.DataFrame()
    
    # FIX: Use full_table() instead of hardcoded string
    table_id = full_table("market_data")

    ticker_list = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT
            ticker,
            DATE(date) AS date,
            open,
            high,
            low,
            close,
            volume,
            LOG(close / NULLIF(LAG(close) OVER (PARTITION BY ticker ORDER BY date), 0)) AS log_return
        FROM `{table_id}`
        WHERE ticker IN ({ticker_list})
        AND DATE(date) >= '{start_date}'
        AND DATE(date) <= '{end_date}'
        AND close > 0
        ORDER BY ticker, DATE(date)
    """
    df = client.query(query).to_dataframe()
    logger.info(
        f"Loaded {len(df):,} price rows for {len(tickers)} tickers "
        f"({start_date}→{end_date})"
    )
    return df


# ── Fama-French Factors ───────────────────────────────────────────────────────

def read_ff_factors(
    start_date: date = None,
    end_date: date = None,
) -> pd.DataFrame:
    """
    Read Fama-French factor returns for a date range.

    Returns
    -------
    DataFrame with: date, mkt_rf, smb, hml, rmw, cma, rf
    """
    client = get_client()

    conditions = []
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT date, mkt_rf, smb, hml, rmw, cma, rf
        FROM `{full_table('ff_factors')}`
        {where_clause}
        ORDER BY date
    """
    df = client.query(query).to_dataframe()
    return df


# ── OOS Strategy Returns ──────────────────────────────────────────────────────

def read_oos_strategy_returns(
    ticker_i: str = None,
    ticker_j: str = None,
) -> pd.DataFrame:
    """
    Read OOS strategy returns — optionally filtered to a specific pair.

    Returns
    -------
    DataFrame with: ticker_i, ticker_j, window_start, lag, oos_date,
                    signal, position, strategy_return_gross, strategy_return_net
    """
    client = get_client()

    conditions = []
    if ticker_i:
        conditions.append(f"ticker_i = '{ticker_i}'")
    if ticker_j:
        conditions.append(f"ticker_j = '{ticker_j}'")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT ticker_i, ticker_j, window_start, lag, oos_date,
               signal, position, strategy_return_gross, strategy_return_net
        FROM `{full_table('oos_strategy_returns')}`
        {where_clause}
        ORDER BY ticker_i, ticker_j, oos_date
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} OOS return records")
    return df


# ── Stability Metrics ─────────────────────────────────────────────────────────

def read_stability_metrics() -> pd.DataFrame:
    """
    Read the latest stability metrics for all pairs.

    Returns
    -------
    DataFrame with: ticker_i, ticker_j, best_lag, mean_dcor, variance_dcor,
                    frequency, half_life, half_life_r2, sharpness, n_windows_observed
    """
    client = get_client()
    query = f"""
        SELECT ticker_i, ticker_j, best_lag, mean_dcor, variance_dcor,
               frequency, half_life, half_life_r2, half_life_stable,
               sharpness, n_windows_observed, last_updated
        FROM `{full_table('stability_metrics')}`
        ORDER BY ticker_i, ticker_j
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} stability metric rows")
    return df


# ── Model Weights ─────────────────────────────────────────────────────────────

def read_model_weights(version: str = None) -> pd.DataFrame:
    """
    Read model β weights. Returns current weights by default.

    Parameters
    ----------
    version : specific model version string, or None for current

    Returns
    -------
    DataFrame with: model_version, refit_date, feature, weight,
                    ci_lower, ci_upper, r2, f_statistic, n_pairs, is_current
    """
    client = get_client()

    if version:
        where = f"WHERE model_version = '{version}'"
    else:
        where = "WHERE is_current = TRUE"

    query = f"""
        SELECT model_version, refit_date, feature, weight,
               ci_lower, ci_upper, r2, f_statistic, n_pairs, is_current
        FROM `{full_table('model_weights')}`
        {where}
        ORDER BY feature
    """
    df = client.query(query).to_dataframe()
    return df


def mark_model_weights_current(model_version: str) -> None:
    """
    Set is_current = TRUE for model_version, FALSE for all others.
    Pass "__none__" to mark all as not current.
    """
    client = get_client()
    table_id = full_table("model_weights")

    if model_version == "__none__":
        query = f"""
            UPDATE `{table_id}`
            SET is_current = FALSE
            WHERE TRUE
        """
    else:
        query = f"""
            UPDATE `{table_id}`
            SET is_current = (model_version = '{model_version}')
            WHERE TRUE
        """

    try:
        client.query(query).result()
        logger.info(f"Marked model weights current: {model_version}")
    except Exception as e:
        logger.error(f"Failed to update model weights current flag: {e}")
        raise


# ── Final Network ─────────────────────────────────────────────────────────────

def upsert_final_network(df: pd.DataFrame) -> None:
    """
    Write final network to BigQuery.
    Deletes existing rows for as_of_date before inserting (idempotent).

    Parameters
    ----------
    df : DataFrame matching final_network schema
    """
    if df.empty:
        logger.warning("upsert_final_network called with empty DataFrame. Skipping.")
        return

    client = get_client()
    table_id = full_table("final_network")

    # Get the as_of_date from the DataFrame
    if "as_of_date" in df.columns:
        as_of_dates = df["as_of_date"].unique()
        for as_of in as_of_dates:
            delete_query = f"""
                DELETE FROM `{table_id}`
                WHERE as_of_date = '{as_of}'
            """
            try:
                client.query(delete_query).result()
            except Exception as e:
                logger.warning(f"Could not delete existing final_network rows for {as_of}: {e}")

    write_dataframe(df, "final_network", write_disposition="WRITE_APPEND")
    logger.info(f"Upserted {len(df):,} rows to final_network")


# ── Logging ───────────────────────────────────────────────────────────────────

def log_pipeline_run(run_info: dict) -> None:
    """
    Append a pipeline run record to pipeline_run_log.

    Parameters
    ----------
    run_info : dict with keys matching pipeline_run_log schema:
        run_id, run_date, window_start, window_end,
        n_pairs_processed, n_significant_pairs, tier3_fraction,
        cpu_hours_used, status, error_message, duration_seconds
    """
    try:
        df = pd.DataFrame([run_info])
        write_dataframe(df, "pipeline_run_log", write_disposition="WRITE_APPEND")
    except Exception as e:
        # Logging failures should never crash the main pipeline
        logger.error(f"Failed to log pipeline run: {e}")


def log_synthetic_health(result: dict) -> None:
    """
    Append a synthetic health check result to synthetic_health_log.

    Parameters
    ----------
    result : dict with keys matching synthetic_health_log schema
    """
    try:
        df = pd.DataFrame([result])
        write_dataframe(df, "synthetic_health_log", write_disposition="WRITE_APPEND")
        logger.info("Synthetic health check result logged.")
    except Exception as e:
        logger.error(f"Failed to log synthetic health check: {e}")