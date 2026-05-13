"""
data.py
-------
Fetches 252 trading days of daily log returns for a given list of tickers.

Primary source : BigQuery `yfinance_stocks_data.market_data`
                 (pre-computed log_return column, same data the pipeline uses)
Fallback source: yfinance — used only for tickers absent from BigQuery
                 (e.g. small-caps below the pipeline universe filters, or
                 recently listed stocks not yet in the table)

The BigQuery path requires Application Default Credentials (ADC), which are
already configured for this project via the attached GCP service account.
"""

import datetime
import logging
import pathlib

import numpy as np
import pandas as pd
import yaml
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 252
# Calendar day buffer ensures we capture exactly LOOKBACK_DAYS trading days
# even across holiday-heavy periods (~1.6× gives a comfortable margin)
_CALENDAR_BUFFER_FACTOR = 1.6

# ---------------------------------------------------------------------------
# Config — read project / dataset / table from the shared config.yaml
# ---------------------------------------------------------------------------

def _load_gcp_config() -> dict:
    """
    Load GCP connection settings from Algorithm/config/config.yaml.
    Falls back to hardcoded defaults if the file cannot be read.
    """
    config_path = (
        pathlib.Path(__file__).parent.parent
        / "Algorithm" / "config" / "config.yaml"
    )
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        gcp = cfg["gcp"]
        return {
            "project_id":     gcp["project_id"],
            "source_dataset": gcp.get("source_dataset", "yfinance_stocks_data"),
            "table":          cfg["tables"].get("market_data", "market_data"),
        }
    except Exception as exc:
        logger.warning(f"Could not read config.yaml ({exc}); using defaults.")
        return {
            "project_id":     "capstone-487001",
            "source_dataset": "yfinance_stocks_data",
            "table":          "market_data",
        }


_GCP = _load_gcp_config()
_FULL_TABLE = (
    f"{_GCP['project_id']}.{_GCP['source_dataset']}.{_GCP['table']}"
)


# ---------------------------------------------------------------------------
# BigQuery fetch
# ---------------------------------------------------------------------------

def _fetch_from_bigquery(
    tickers: list,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Query the last `lookback_days` trading days of log_return from BigQuery
    for the requested tickers.

    Returns a wide DataFrame: index = date, columns = tickers.
    Tickers with no rows in the table are simply absent from the result.
    """
    if not tickers:
        return pd.DataFrame()

    ticker_list = ", ".join(f"'{t}'" for t in tickers)
    # Approximate calendar-day lookback with a buffer for holidays/weekends
    calendar_days = int(lookback_days * _CALENDAR_BUFFER_FACTOR) + 30

    query = f"""
        WITH recent AS (
            SELECT
                DATE(date)   AS date,
                ticker,
                log_return
            FROM `{_FULL_TABLE}`
            WHERE ticker IN ({ticker_list})
              AND date >= DATE_SUB(CURRENT_DATE(), INTERVAL {calendar_days} DAY)
              AND log_return IS NOT NULL
        ),
        ranked AS (
            SELECT
                date,
                ticker,
                log_return,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker
                    ORDER BY date DESC
                ) AS rn
            FROM recent
        )
        SELECT date, ticker, log_return
        FROM ranked
        WHERE rn <= {lookback_days}
        ORDER BY ticker, date
    """

    try:
        client = bigquery.Client(project=_GCP["project_id"])
        df = client.query(query).to_dataframe()
    except GoogleCloudError as exc:
        logger.warning(f"BigQuery query failed: {exc}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="ticker", values="log_return")
    wide.sort_index(inplace=True)
    return wide


# ---------------------------------------------------------------------------
# yfinance fallback
# ---------------------------------------------------------------------------

def _fetch_from_yfinance(
    tickers: list,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Download prices from yfinance and compute log returns.
    Used only for tickers that BigQuery could not supply.
    """
    import yfinance as yf

    if not tickers:
        return pd.DataFrame()

    end = datetime.date.today()
    calendar_days = int(lookback_days * _CALENDAR_BUFFER_FACTOR) + 30
    start = end - datetime.timedelta(days=calendar_days)

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(raw, pd.Series):
        name = tickers[0] if isinstance(tickers, list) else tickers
        raw = raw.to_frame(name=name)

    available = [t for t in tickers if t in raw.columns]
    if not available:
        return pd.DataFrame()

    prices = raw[available].ffill().dropna()
    returns = np.log(prices / prices.shift(1)).dropna()

    # Trim to lookback_days
    if len(returns) > lookback_days:
        returns = returns.iloc[-lookback_days:]

    returns.index = pd.to_datetime(returns.index)
    return returns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_returns(
    tickers: list,
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple:
    """
    Fetch `lookback_days` of daily log returns for the requested tickers.

    Tries BigQuery first for every ticker. Any ticker that comes back with
    fewer than 60 rows (insufficient history) is retried via yfinance.
    Tickers that fail both sources are reported in `missing`.

    Parameters
    ----------
    tickers       : list of ticker symbols
    lookback_days : number of trading days to include (default 252)

    Returns
    -------
    returns   : pd.DataFrame, shape (≤lookback_days, n_available), log returns
    available : list of tickers successfully loaded
    missing   : list of tickers that could not be loaded from either source
    """
    tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    # --- Primary: BigQuery ---
    logger.info(f"Fetching {len(tickers)} tickers from BigQuery...")
    bq_wide = _fetch_from_bigquery(tickers, lookback_days)

    bq_ok = []
    needs_fallback = []

    for t in tickers:
        if t in bq_wide.columns and bq_wide[t].count() >= 60:
            bq_ok.append(t)
        else:
            needs_fallback.append(t)

    if needs_fallback:
        logger.info(
            f"{len(needs_fallback)} ticker(s) not in BigQuery, "
            f"falling back to yfinance: {needs_fallback}"
        )

    # --- Fallback: yfinance ---
    yf_wide = _fetch_from_yfinance(needs_fallback, lookback_days)

    yf_ok = []
    missing = []
    for t in needs_fallback:
        if t in yf_wide.columns and yf_wide[t].count() >= 60:
            yf_ok.append(t)
        else:
            missing.append(t)

    if missing:
        logger.warning(f"Could not load returns for: {missing}")

    # --- Merge BigQuery and yfinance results ---
    frames = []
    if bq_ok:
        frames.append(bq_wide[bq_ok])
    if yf_ok:
        frames.append(yf_wide[yf_ok])

    if not frames:
        raise ValueError(
            f"No return data could be loaded for any of: {tickers}"
        )

    if len(frames) == 1:
        combined = frames[0]
    else:
        combined = pd.concat(frames, axis=1).sort_index()

    # Drop rows where all columns are NaN, forward-fill isolated gaps
    combined = combined.ffill().dropna(how="all")

    # Trim to exactly lookback_days rows
    if len(combined) > lookback_days:
        combined = combined.iloc[-lookback_days:]

    available = bq_ok + yf_ok
    return combined[available], available, missing