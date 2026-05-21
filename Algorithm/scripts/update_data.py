"""
scripts/update_data.py
----------------------
Incremental market data update script for LagLens.

Steps
-----
1. market_data update  — downloads new OHLCV from yfinance for all tickers
   already present in market_data, from each ticker's last stored date to today.
2. ticker_metadata refresh — re-fetches sector, market cap, etc. via yfinance.
3. (Optional) general_market_data update — same incremental OHLCV logic but for
   the EXTRA_TICKERS list below.  Only runs when --update-general is passed.

Usage
-----
    # From the Algorithm/ directory:
    python -m scripts.update_data                   # steps 1 + 2
    python -m scripts.update_data --update-general  # steps 1 + 2 + 3
    python -m scripts.update_data --dry-run         # preview only, no writes

Prerequisites
-------------
    pip install yfinance google-cloud-bigquery pandas numpy pyarrow

Credentials
-----------
    Reads GOOGLE_APPLICATION_CREDENTIALS env var (preferred), or falls back to
    ../backend/secrets/gcp-service-account.json relative to this script's
    Algorithm/ parent directory.
"""

import argparse
import logging
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from google.cloud import bigquery
from google.oauth2 import service_account

# ── Logging ───────────────────────────────────────────────────────────────────

_log_filename = f"update_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_filename),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GCP_PROJECT      = "capstone-487001"
SOURCE_DATASET   = "yfinance_stocks_data"   # read-only source (ticker discovery fallback)
WRITE_DATASET    = "output_results"          # where the backend reads; service account has write access here
MARKET_DATA_TABLE    = f"{GCP_PROJECT}.{WRITE_DATASET}.market_data"
TICKER_META_TABLE    = f"{GCP_PROJECT}.{WRITE_DATASET}.ticker_metadata"
GENERAL_DATA_TABLE   = f"{GCP_PROJECT}.{WRITE_DATASET}.general_market_data"

BATCH_SIZE = 50   # tickers per yfinance download call

# Hardcoded fallback — used only if all dynamic sources fail.
_FALLBACK_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "GLD", "TLT",
    "XLF", "XLK", "XLE", "XLV", "ARKK",
    "TSM", "BABA", "NIO", "ASML", "SHOP", "SE", "SPOT", "MELI",
    "HOOD", "COIN", "RIVN", "SOFI", "RBLX", "SNAP", "PINS", "DUOL",
    "MSTR", "RIOT", "MARA", "ARM",  "CART",
    "GME",  "AMC",  "TLRY", "CGC",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    """
    Convert val to float, returning None for anything non-finite.
    Handles yfinance quirks like the string "Infinity" for P/E ratios
    on zero-earnings stocks.
    """
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return None


# ── Dynamic ticker fetching ───────────────────────────────────────────────────

def _fetch_sp_index_tickers() -> list[str]:
    """
    Scrape S&P 500, S&P 400 (mid-cap), and S&P 600 (small-cap) constituent
    lists from Wikipedia using pandas.read_html.

    Combined this yields ~1,500 well-known tickers.  Many S&P 500 stocks will
    already be in market_data (top 2000 R3000 includes them); the S&P 400 and
    S&P 600 are where genuinely new stocks will come from.

    Returns a deduplicated sorted list of ticker symbols.
    """
    sources = [
        # (URL, column name in the Wikipedia table)
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Ticker"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Ticker"),
    ]

    tickers: set[str] = set()
    for url, col in sources:
        try:
            tables = pd.read_html(url, attrs={"id": "constituents"})
            if not tables:
                tables = pd.read_html(url)
            df = tables[0]
            # Try the expected column name, then fall back to any column that
            # looks like it contains ticker symbols
            if col in df.columns:
                raw = df[col].dropna().astype(str).tolist()
            else:
                # Find the column with the most 1-5 letter uppercase entries
                best_col, best_count = None, 0
                for c in df.columns:
                    count = df[c].astype(str).str.match(r"^[A-Z]{1,5}$").sum()
                    if count > best_count:
                        best_count, best_col = count, c
                raw = df[best_col].dropna().astype(str).tolist() if best_col else []

            # Normalise: BRK.B → BRK-B, strip whitespace, keep 1-5 letter only
            cleaned = [
                t.strip().replace(".", "-").upper()
                for t in raw
                if re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", t.strip().upper())
            ]
            before = len(tickers)
            tickers.update(cleaned)
            logger.info(f"  Wikipedia {url.split('List_of_')[-1]}: "
                        f"{len(cleaned)} tickers ({len(tickers) - before} new)")
        except Exception as e:
            logger.warning(f"  Could not scrape {url}: {e}")

    return sorted(tickers)


def _fetch_nasdaq_trader_tickers() -> list[str]:
    """
    Download all US-listed stock tickers from NASDAQ Trader's public symbol
    directory (no authentication required).

    Sources
    -------
    nasdaqlisted.txt  — all NASDAQ-listed securities
    otherlisted.txt   — all NYSE / NYSE American / BATS-listed securities

    Filters applied
    ---------------
    • Test issues removed (Test Issue == 'Y')
    • Only clean tickers: 1–5 uppercase letters (removes warrants, rights,
      preferred shares, units that carry suffixes like .WS, -WT, +, ^)

    Returns ~6 000–8 000 deduplicated tickers.
    """
    import re as _re

    NASDAQ_URL = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    OTHER_URL  = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"

    configs = [
        (NASDAQ_URL, "Symbol",     "Test Issue"),
        (OTHER_URL,  "ACT Symbol", "Test Issue"),
    ]

    tickers: set[str] = set()
    for url, sym_col, test_col in configs:
        try:
            df = pd.read_csv(url, sep="|")
            # Drop the trailing metadata row NASDAQ appends
            df = df[~df[sym_col].astype(str).str.startswith("File Creation")]
            # Remove test issues
            if test_col in df.columns:
                df = df[df[test_col].astype(str).str.strip() == "N"]
            # Keep only clean 1–5 letter tickers (common stocks + ETFs)
            valid_mask = df[sym_col].astype(str).str.match(r"^[A-Z]{1,5}$")
            batch = df.loc[valid_mask, sym_col].astype(str).tolist()
            before = len(tickers)
            tickers.update(batch)
            logger.info(f"  NASDAQ Trader {url.split('/')[-1]}: "
                        f"{len(batch)} tickers ({len(tickers) - before} new)")
        except Exception as e:
            logger.warning(f"  Could not fetch {url}: {e}")

    return sorted(tickers)


def _get_extended_ticker_list(use_all_listed: bool = False) -> list[str]:
    """
    Build the candidate list for general_market_data.

    Strategy (with automatic fallback):
      1. Wikipedia S&P 500 + S&P 400 + S&P 600  (~1 500 tickers, curated)
      2. NASDAQ Trader full listing              (~7 000 tickers, exhaustive)
         — only used when use_all_listed=True or Wikipedia scraping fails
      3. _FALLBACK_TICKERS hardcoded list        (last resort)

    Tickers already in market_data are filtered out in update_general_stocks().
    """
    import re  # noqa: F811 — already imported at module level but needed here too

    logger.info("Building extended ticker list...")

    tickers: list[str] = []

    if use_all_listed:
        logger.info("  Mode: NASDAQ Trader (all US-listed stocks)")
        tickers = _fetch_nasdaq_trader_tickers()
    else:
        logger.info("  Mode: Wikipedia S&P 500 + S&P 400 + S&P 600")
        tickers = _fetch_sp_index_tickers()

    if not tickers:
        logger.warning("  Dynamic fetch returned nothing — falling back to hardcoded list")
        tickers = list(_FALLBACK_TICKERS)

    logger.info(f"  Extended ticker list: {len(tickers):,} candidates total")
    return tickers

# ── Credentials ───────────────────────────────────────────────────────────────

def _build_client() -> bigquery.Client:
    """
    Build a BigQuery client.
    1. Uses GOOGLE_APPLICATION_CREDENTIALS if set.
    2. Falls back to ../backend/secrets/gcp-service-account.json.
    3. Falls back to Application Default Credentials (gcloud auth).
    """
    # Option 1 – explicit env var
    creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_env and os.path.isfile(creds_env):
        logger.info(f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {creds_env}")
        creds = service_account.Credentials.from_service_account_file(creds_env)
        return bigquery.Client(project=GCP_PROJECT, credentials=creds)

    # Option 2 – sibling secrets folder (run from Algorithm/ directory)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _algo_dir   = os.path.dirname(_script_dir)
    sa_path = os.path.join(_algo_dir, "..", "backend", "secrets", "gcp-service-account.json")
    sa_path = os.path.normpath(sa_path)
    if os.path.isfile(sa_path):
        logger.info(f"Using service account: {sa_path}")
        creds = service_account.Credentials.from_service_account_file(sa_path)
        return bigquery.Client(project=GCP_PROJECT, credentials=creds)

    # Option 3 – application default credentials
    logger.warning("No project ID could be determined. Consider running "
                   "`gcloud config set project` or setting GOOGLE_CLOUD_PROJECT")
    return bigquery.Client(project=GCP_PROJECT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _batched(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _to_date(val) -> date:
    """Coerce a BQ DATETIME / DATE / string to a Python date."""
    if isinstance(val, date):
        return val if not isinstance(val, datetime) else val.date()
    return pd.Timestamp(val).date()


# ── Step 1 & 3: OHLCV download + append ──────────────────────────────────────

def _get_latest_dates(client: bigquery.Client, table_id: str) -> dict[str, date]:
    """
    Return {ticker: latest_date} for all tickers currently in table_id.
    Used to determine the incremental download window.
    """
    query = f"""
        SELECT ticker, MAX(DATE(date)) AS latest_date
        FROM `{table_id}`
        GROUP BY ticker
    """
    df = client.query(query).to_dataframe()
    return {row.ticker: _to_date(row.latest_date) for _, row in df.iterrows()}


def _download_batch(tickers: list[str], start: date, end: date, max_retries: int = 3) -> pd.DataFrame:
    """
    Download OHLCV for a batch of tickers via yfinance.
    Retries up to max_retries times with linear back-off (60s × attempt) on
    rate-limit errors, then gives up and returns an empty DataFrame.

    Returns a tidy DataFrame with columns:
        ticker, date, open, high, low, close, adj_close, volume, log_return
    """
    ticker_str = " ".join(tickers)
    raw = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                ticker_str,
                start=str(start),
                end=str(end + timedelta(days=1)),   # end is exclusive in yfinance
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            break  # success — exit retry loop
        except Exception as e:
            err_str  = str(e)
            exc_type = type(e).__name__
            is_rate_limit = (
                "RateLimit" in exc_type
                or "Too Many Requests" in err_str
                or "rate limit" in err_str.lower()
            )
            if is_rate_limit:
                wait_sec = 60 * attempt   # 60 s, 120 s, 180 s
                if attempt < max_retries:
                    logger.warning(
                        f"  Rate limited (attempt {attempt}/{max_retries}). "
                        f"Waiting {wait_sec}s before retry..."
                    )
                    time.sleep(wait_sec)
                else:
                    logger.warning(
                        f"  Giving up batch after {max_retries} rate-limit retries: {e}"
                    )
                    return pd.DataFrame()
            else:
                logger.warning(f"  yfinance download failed for batch: {e}")
                return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex columns (field, ticker) when >1 ticker
    if len(tickers) == 1:
        raw.columns = [c.lower() for c in raw.columns]
        raw["ticker"] = tickers[0]
        raw = raw.reset_index()
        raw = raw.rename(columns={"Date": "date", "Adj Close": "adj_close"})
    else:
        raw.columns = [f"{col[0].lower()}_{col[1]}" for col in raw.columns]
        raw = raw.reset_index()
        frames = []
        for tkr in tickers:
            cols = {
                "Date":                  "date",
                f"open_{tkr}":           "open",
                f"high_{tkr}":           "high",
                f"low_{tkr}":            "low",
                f"close_{tkr}":          "close",
                f"adj close_{tkr}":      "adj_close",
                f"volume_{tkr}":         "volume",
            }
            available = {k: v for k, v in cols.items() if k in raw.columns}
            if "date" not in available:
                available["Date"] = "date"
            sub = raw[list(available.keys())].rename(columns=available).copy()
            sub["ticker"] = tkr
            frames.append(sub)
        raw = pd.concat(frames, ignore_index=True)

    # Normalise
    raw["date"] = pd.to_datetime(raw["date"])
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in raw.columns:
            raw[col] = float("nan")
    raw = raw.dropna(subset=["close"])
    raw["volume"] = raw["volume"].fillna(0).astype("int64")

    # log_return per ticker (sorted by date within each ticker)
    raw = raw.sort_values(["ticker", "date"])
    raw["log_return"] = (
        raw.groupby("ticker")["close"]
        .transform(lambda s: np.log(s / s.shift(1)))
    )

    return raw[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "log_return"]]


def _upload_df(client: bigquery.Client, df: pd.DataFrame, table_id: str, dry_run: bool) -> int:
    """Append df to table_id. Returns rows written (0 in dry-run)."""
    if df.empty:
        return 0
    if dry_run:
        logger.info(f"    [DRY RUN] Would write {len(df):,} rows to {table_id}")
        return 0

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=False,
        schema=[
            bigquery.SchemaField("ticker",    "STRING"),
            bigquery.SchemaField("date",      "DATETIME"),
            bigquery.SchemaField("open",      "FLOAT64"),
            bigquery.SchemaField("high",      "FLOAT64"),
            bigquery.SchemaField("low",       "FLOAT64"),
            bigquery.SchemaField("close",     "FLOAT64"),
            bigquery.SchemaField("adj_close", "FLOAT64"),
            bigquery.SchemaField("volume",    "INT64"),
            bigquery.SchemaField("log_return","FLOAT64"),
        ],
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    return len(df)


def update_market_data(
    client: bigquery.Client,
    table_id: str,
    dry_run: bool = False,
    extra_tickers: list[str] | None = None,
) -> tuple[int, int]:
    """
    Incremental OHLCV update for table_id.
    - Reads latest date per ticker from table_id.
    - Downloads new data from (latest + 1 day) to today.
    - Appends to table_id.

    extra_tickers: if provided, these tickers are added to the download even if
                   they don't yet exist in the table (first-time ingest).

    Returns (rows_written, tickers_updated).
    """
    today = date.today()
    logger.info(f"Querying latest dates from {table_id}...")
    latest_dates = _get_latest_dates(client, table_id)
    logger.info(f"  Found {len(latest_dates):,} tickers in {table_id}")

    # Merge in any extra tickers not yet in the table (start from 5 years ago)
    if extra_tickers:
        five_years_ago = today - timedelta(days=5 * 365)
        new_tickers = {t: five_years_ago for t in extra_tickers if t not in latest_dates}
        if new_tickers:
            logger.info(f"  Adding {len(new_tickers)} new tickers: {list(new_tickers.keys())}")
            latest_dates.update(new_tickers)

    # Group tickers by their start date to minimise yfinance calls
    # (most tickers share the same latest_date → one large batch)
    from collections import defaultdict
    by_start: dict[date, list[str]] = defaultdict(list)
    skipped = 0
    for tkr, last_dt in latest_dates.items():
        start = last_dt + timedelta(days=1)
        if start > today:
            skipped += 1
            continue
        by_start[start].append(tkr)

    if not by_start:
        logger.info(f"  All {skipped} tickers are already up to date.")
        return 0, 0

    all_starts = sorted(by_start.keys())
    earliest = all_starts[0]
    latest_start = all_starts[-1]
    logger.info(
        f"  {sum(len(v) for v in by_start.values()):,} tickers need updating "
        f"(start range: {earliest} → {latest_start}, {skipped} already up to date)"
    )

    total_rows = 0
    total_tickers = 0

    # For simplicity, download all tickers from the EARLIEST common start.
    # For tickers with a later start, extra rows will be filtered out client-side.
    all_tickers = [t for tickers in by_start.values() for t in tickers]
    logger.info(f"  Downloading {len(all_tickers)} tickers from {earliest} → {today}")

    batches = list(_batched(all_tickers, BATCH_SIZE))
    for i, batch in enumerate(batches, 1):
        logger.info(f"    Batch {i}/{len(batches)}: {len(batch)} tickers")
        df = _download_batch(batch, earliest, today)

        if df.empty:
            logger.warning(f"    No data returned for batch {i}")
            time.sleep(2)
            continue

        # Filter out rows each ticker already has
        filtered_rows = []
        for tkr, grp in df.groupby("ticker"):
            cutoff = latest_dates.get(tkr, date.min)
            new_rows = grp[grp["date"].dt.date > cutoff]
            if not new_rows.empty:
                filtered_rows.append(new_rows)
                total_tickers += 1

        if not filtered_rows:
            logger.info(f"    No new rows after filtering for batch {i}")
            time.sleep(2)
            continue

        batch_df = pd.concat(filtered_rows, ignore_index=True)
        rows_written = _upload_df(client, batch_df, table_id, dry_run)
        total_rows += rows_written
        if not dry_run:
            logger.info(f"    Wrote {rows_written:,} rows for {len(filtered_rows)} tickers")
        time.sleep(2)   # be polite to yfinance

    return total_rows, total_tickers


# ── Step 2: ticker_metadata refresh ──────────────────────────────────────────

def _fetch_yf_metadata(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch metadata for a list of tickers via yfinance .info.
    Returns DataFrame with columns matching the actual ticker_metadata schema:
        ticker, company_name, sector, industry, market_cap, pe_ratio,
        country, exchange, currency, website, description, employees,
        last_updated, is_active
    """
    now = datetime.now()
    records = []
    for i, tkr in enumerate(tickers, 1):
        if i % 50 == 0:
            logger.info(f"  Metadata: {i}/{len(tickers)} fetched...")
        try:
            info = yf.Ticker(tkr).info
            records.append({
                "ticker":       tkr,
                "company_name": info.get("longName") or info.get("shortName") or tkr,
                "sector":       info.get("sector"),
                "industry":     info.get("industry"),
                "market_cap":   info.get("marketCap"),
                "pe_ratio":     _safe_float(info.get("trailingPE")),
                "country":      info.get("country"),
                "exchange":     info.get("exchange"),
                "currency":     info.get("currency"),
                "website":      info.get("website"),
                "description":  info.get("longBusinessSummary"),
                "employees":    info.get("fullTimeEmployees"),
                "last_updated": now,
                "is_active":    True,
            })
        except Exception as e:
            logger.debug(f"  Could not fetch metadata for {tkr}: {e}")
            records.append({
                "ticker":       tkr,
                "company_name": tkr,
                "sector":       None,
                "industry":     None,
                "market_cap":   None,
                "pe_ratio":     None,
                "country":      None,
                "exchange":     None,
                "currency":     None,
                "website":      None,
                "description":  None,
                "employees":    None,
                "last_updated": now,
                "is_active":    False,
            })
    return pd.DataFrame(records)


def refresh_ticker_metadata(
    client: bigquery.Client,
    dry_run: bool = False,
) -> int:
    """
    Re-fetch metadata for all tickers in market_data and upsert into
    ticker_metadata using a MERGE statement.

    Returns number of tickers processed.
    """
    logger.info(f"Fetching ticker list from {MARKET_DATA_TABLE}...")
    query = f"SELECT DISTINCT ticker FROM `{MARKET_DATA_TABLE}` ORDER BY ticker"
    tickers = [row.ticker for row in client.query(query).result()]
    logger.info(f"  Fetching metadata for {len(tickers):,} tickers from Yahoo Finance...")

    meta_df = _fetch_yf_metadata(tickers)
    logger.info(f"  Fetched {len(meta_df):,} metadata rows")

    # INTEGER columns: use pandas nullable Int64 so None/NaN stays null (not NaN→crash)
    meta_df["market_cap"] = meta_df["market_cap"].astype("Int64")
    meta_df["employees"]  = meta_df["employees"].astype("Int64")

    if dry_run:
        logger.info(f"  [DRY RUN] Would upsert {len(meta_df):,} rows into {TICKER_META_TABLE}")
        return len(meta_df)

    # Write to a staging table, then MERGE into ticker_metadata
    staging_id = f"{GCP_PROJECT}.{WRITE_DATASET}._meta_staging_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta_schema = [
        bigquery.SchemaField("ticker",       "STRING"),
        bigquery.SchemaField("company_name", "STRING"),
        bigquery.SchemaField("sector",       "STRING"),
        bigquery.SchemaField("industry",     "STRING"),
        bigquery.SchemaField("market_cap",   "INT64"),
        bigquery.SchemaField("pe_ratio",     "FLOAT64"),
        bigquery.SchemaField("country",      "STRING"),
        bigquery.SchemaField("exchange",     "STRING"),
        bigquery.SchemaField("currency",     "STRING"),
        bigquery.SchemaField("website",      "STRING"),
        bigquery.SchemaField("description",  "STRING"),
        bigquery.SchemaField("employees",    "INT64"),
        bigquery.SchemaField("last_updated", "TIMESTAMP"),
        bigquery.SchemaField("is_active",    "BOOL"),
    ]
    table = bigquery.Table(staging_id, schema=meta_schema)
    client.create_table(table, exists_ok=True)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=meta_schema,
    )
    job = client.load_table_from_dataframe(meta_df, staging_id, job_config=job_config)
    job.result()

    merge_sql = f"""
        MERGE `{TICKER_META_TABLE}` T
        USING `{staging_id}` S
        ON T.ticker = S.ticker
        WHEN MATCHED THEN UPDATE SET
            T.company_name = S.company_name,
            T.sector       = S.sector,
            T.industry     = S.industry,
            T.market_cap   = S.market_cap,
            T.pe_ratio     = S.pe_ratio,
            T.country      = S.country,
            T.exchange     = S.exchange,
            T.currency     = S.currency,
            T.website      = S.website,
            T.description  = S.description,
            T.employees    = S.employees,
            T.last_updated = S.last_updated,
            T.is_active    = S.is_active
        WHEN NOT MATCHED THEN INSERT
            (ticker, company_name, sector, industry, market_cap, pe_ratio,
             country, exchange, currency, website, description, employees,
             last_updated, is_active)
            VALUES
            (S.ticker, S.company_name, S.sector, S.industry, S.market_cap, S.pe_ratio,
             S.country, S.exchange, S.currency, S.website, S.description, S.employees,
             S.last_updated, S.is_active)
    """
    client.query(merge_sql).result()

    # Clean up staging
    try:
        client.delete_table(staging_id)
    except Exception:
        pass

    return len(meta_df)


# ── Step 3: general_market_data ───────────────────────────────────────────────

def ensure_general_market_data_table(client: bigquery.Client) -> None:
    """
    Create general_market_data if it doesn't exist yet.
    Uses the same schema as market_data.
    """
    schema = [
        bigquery.SchemaField("ticker",     "STRING"),
        bigquery.SchemaField("date",       "DATETIME"),
        bigquery.SchemaField("open",       "FLOAT64"),
        bigquery.SchemaField("high",       "FLOAT64"),
        bigquery.SchemaField("low",        "FLOAT64"),
        bigquery.SchemaField("close",      "FLOAT64"),
        bigquery.SchemaField("adj_close",  "FLOAT64"),
        bigquery.SchemaField("volume",     "INT64"),
        bigquery.SchemaField("log_return", "FLOAT64"),
    ]
    table = bigquery.Table(GENERAL_DATA_TABLE, schema=schema)
    table.description = (
        "OHLCV for general stocks outside the lead-lag analysis universe. "
        "Used by the frontend for basic stock display. "
        "NEVER read by the Algorithm pipeline."
    )
    # Partition by date for efficient range queries
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY, field="date"
    )
    table.clustering_fields = ["ticker"]
    client.create_table(table, exists_ok=True)
    logger.info(f"  ✓ Table ready: {GENERAL_DATA_TABLE}")


# yfinance symbol → BQ/frontend symbol for the three dashboard indices
_INDEX_SYMBOL_MAP = {
    "^GSPC": "SPX",   # S&P 500
    "^IXIC": "IXIC",  # NASDAQ Composite
    "^DJI":  "DJI",   # Dow Jones Industrial Average
}


def seed_indices(
    client: bigquery.Client,
    dry_run: bool = False,
) -> int:
    """
    Download full OHLCV history for the three dashboard indices (S&P 500,
    NASDAQ, Dow Jones) and write them into general_market_data.

    yfinance symbols (^GSPC, ^IXIC, ^DJI) are remapped to the clean names
    the frontend expects (SPX, IXIC, DJI) before writing.

    Only downloads rows newer than the latest date already in the table for
    each index (incremental-safe).

    Returns the number of rows written.
    """
    ensure_general_market_data_table(client)

    yf_symbols  = list(_INDEX_SYMBOL_MAP.keys())   # ["^GSPC", "^IXIC", "^DJI"]
    bq_symbols  = list(_INDEX_SYMBOL_MAP.values())  # ["SPX",   "IXIC",  "DJI"]

    # Find the latest date already stored for each index
    try:
        existing = _get_latest_dates(client, GENERAL_DATA_TABLE)
    except Exception:
        existing = {}

    today = date.today()
    five_years_ago = today - timedelta(days=5 * 365)

    # Determine per-index start date
    starts: dict[str, date] = {}
    for bq_sym in bq_symbols:
        last = existing.get(bq_sym)
        starts[bq_sym] = (last + timedelta(days=1)) if last else five_years_ago

    earliest_start = min(starts.values())
    if earliest_start > today:
        logger.info("  All indices already up to date.")
        return 0

    logger.info(f"  Downloading indices {bq_symbols} from {earliest_start} → {today}")

    # Download all three in one yfinance call using the ^-prefixed symbols
    raw = yf.download(
        " ".join(yf_symbols),
        start=str(earliest_start),
        end=str(today + timedelta(days=1)),
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw is None or raw.empty:
        logger.warning("  yfinance returned no data for indices.")
        return 0

    frames = []
    for yf_sym, bq_sym in _INDEX_SYMBOL_MAP.items():
        try:
            sub = pd.DataFrame()
            sub["date"]      = raw.index
            sub["open"]      = raw[("Open",      yf_sym)].values
            sub["high"]      = raw[("High",      yf_sym)].values
            sub["low"]       = raw[("Low",       yf_sym)].values
            sub["close"]     = raw[("Close",     yf_sym)].values
            sub["adj_close"] = raw[("Adj Close", yf_sym)].values
            sub["volume"]    = raw[("Volume",    yf_sym)].values
            sub["ticker"]    = bq_sym
        except KeyError:
            logger.warning(f"  {yf_sym} ({bq_sym}) not in yfinance response — skipping (rate limited?)")
            continue

        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.dropna(subset=["close"])
        sub["volume"] = sub["volume"].fillna(0).astype("int64")

        # Filter to only new rows
        cutoff = starts[bq_sym]
        sub = sub[sub["date"].dt.date >= cutoff]

        # log_return
        sub = sub.sort_values("date")
        sub["log_return"] = np.log(sub["close"] / sub["close"].shift(1))

        if not sub.empty:
            frames.append(sub)
            logger.info(f"  {bq_sym}: {len(sub):,} rows ready")

    if not frames:
        logger.info("  No new rows for any index.")
        return 0

    df = pd.concat(frames, ignore_index=True)
    df = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "log_return"]]

    rows_written = _upload_df(client, df, GENERAL_DATA_TABLE, dry_run)
    logger.info(f"  ✓ Indices: {rows_written:,} rows written for {bq_symbols}")
    return rows_written


def update_general_stocks(
    client: bigquery.Client,
    dry_run: bool = False,
    use_all_listed: bool = False,
) -> tuple[int, int]:
    """
    Fetch a large candidate list of tickers, remove any already in market_data,
    and do an incremental OHLCV update into general_market_data for the rest.

    Parameters
    ----------
    use_all_listed : if True, use NASDAQ Trader (~7 000 tickers) instead of
                     Wikipedia S&P indices (~1 500 tickers).  Slower but more
                     comprehensive.
    """
    # ── Step 3a: get the full candidate list ─────────────────────────────────
    candidates = _get_extended_ticker_list(use_all_listed=use_all_listed)

    # ── Step 3b: filter out anything already in market_data ──────────────────
    logger.info("Checking which candidates are already in market_data...")
    universe_query = f"SELECT DISTINCT ticker FROM `{MARKET_DATA_TABLE}`"
    universe_set = {row.ticker for row in client.query(universe_query).result()}

    # Also check what's already in general_market_data (incremental-safe)
    try:
        gen_query = f"SELECT DISTINCT ticker FROM `{GENERAL_DATA_TABLE}`"
        already_general = {row.ticker for row in client.query(gen_query).result()}
    except Exception:
        already_general = set()

    truly_new    = [t for t in candidates if t not in universe_set and t not in already_general]
    already_main = [t for t in candidates if t in universe_set]
    already_gen  = [t for t in candidates if t in already_general]

    logger.info(f"  {len(candidates):,} candidates total")
    logger.info(f"  {len(already_main):,} already in market_data   (skipping)")
    logger.info(f"  {len(already_gen):,}  already in general_market_data (incremental update)")
    logger.info(f"  {len(truly_new):,}  genuinely new → will ingest into general_market_data")

    if dry_run:
        logger.info(f"  [DRY RUN] Would ingest {len(truly_new):,} new tickers. Sample:")
        for t in truly_new[:30]:
            logger.info(f"    {t}")
        if len(truly_new) > 30:
            logger.info(f"    ... and {len(truly_new) - 30} more")
        return 0, len(truly_new)

    if not truly_new and not already_gen:
        logger.info("  Nothing to do — all candidates already covered.")
        ensure_general_market_data_table(client)   # still create table if missing
        return 0, 0

    ensure_general_market_data_table(client)

    # Ingest new tickers + do incremental update for existing general tickers
    all_to_process = truly_new + list(already_gen)
    rows, tickers = update_market_data(
        client,
        table_id=GENERAL_DATA_TABLE,
        dry_run=dry_run,
        extra_tickers=all_to_process,
    )
    return rows, tickers


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    dry_run: bool = False,
    update_general: bool = False,
    use_all_listed: bool = False,
    metadata_only: bool = False,
    seed_indices_only: bool = False,
) -> None:
    logger.info("=" * 60)
    logger.info("DATA UPDATE SCRIPT — LagLens")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if dry_run:
        logger.info("MODE: DRY RUN — no writes will be made")
    logger.info("=" * 60)

    client = _build_client()

    # ── Indices-only shortcut ─────────────────────────────────────────────
    if seed_indices_only:
        logger.info("\n" + "=" * 60)
        logger.info("SEED INDICES: S&P 500 (SPX), NASDAQ (IXIC), Dow Jones (DJI)")
        logger.info("=" * 60)
        rows_idx = seed_indices(client, dry_run=dry_run)
        logger.info(f"\n  Indices seeded: {rows_idx:,} rows written")
        logger.info("\n" + "=" * 60)
        logger.info("DATA UPDATE COMPLETE")
        logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        return

    # ── Step 1: market_data ───────────────────────────────────────────────
    if metadata_only:
        logger.info("\n  Skipping Step 1 (--metadata-only flag set)")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Updating market_data")
        logger.info("=" * 60)
        rows1, tickers1 = update_market_data(client, MARKET_DATA_TABLE, dry_run=dry_run)
        logger.info(
            f"\n  Market data update complete: {rows1:,} rows written, "
            f"{tickers1:,} tickers updated"
        )

    # ── Step 2: ticker_metadata ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Refreshing ticker_metadata")
    logger.info("=" * 60)
    n_meta = refresh_ticker_metadata(client, dry_run=dry_run)
    logger.info(f"\n  Metadata refresh complete: {n_meta:,} tickers processed")

    # ── Step 3: general_market_data (optional) ────────────────────────────
    if update_general:
        mode_label = "NASDAQ Trader (all US-listed)" if use_all_listed else "Wikipedia S&P 500/400/600"
        logger.info("\n" + "=" * 60)
        logger.info(f"STEP 3: Updating general_market_data — source: {mode_label}")
        logger.info("=" * 60)
        rows3, tickers3 = update_general_stocks(
            client, dry_run=dry_run, use_all_listed=use_all_listed
        )
        logger.info(
            f"\n  General stocks update complete: {rows3:,} rows written, "
            f"{tickers3:,} tickers updated/queued"
        )

    logger.info("\n" + "=" * 60)
    logger.info("DATA UPDATE COMPLETE")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incremental LagLens data update",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Update universe + metadata only
  python -m scripts.update_data

  # + general stocks from S&P 500/400/600 (~1 500 candidates)
  python -m scripts.update_data --update-general

  # + general stocks from ALL US-listed tickers (~7 000 candidates, slower)
  python -m scripts.update_data --update-general --all-listed

  # Dry run first to see what would be ingested
  python -m scripts.update_data --dry-run --update-general --all-listed
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview downloads/writes without touching BigQuery.",
    )
    parser.add_argument(
        "--update-general",
        action="store_true",
        help="Also update general_market_data for extra stocks outside the universe.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip Step 1 (OHLCV update) and run only Step 2 (ticker_metadata refresh).",
    )
    parser.add_argument(
        "--seed-indices",
        action="store_true",
        help=(
            "Only download SPX (^GSPC), IXIC (^IXIC), and DJI (^DJI) into "
            "general_market_data. Skips all other steps."
        ),
    )
    parser.add_argument(
        "--all-listed",
        action="store_true",
        help=(
            "Use NASDAQ Trader's full US listing (~7 000 tickers) as the "
            "candidate source instead of Wikipedia S&P indices (~1 500). "
            "More comprehensive but significantly slower to ingest."
        ),
    )
    args = parser.parse_args()
    main(
        dry_run=args.dry_run,
        update_general=args.update_general,
        use_all_listed=args.all_listed,
        metadata_only=args.metadata_only,
        seed_indices_only=args.seed_indices,
    )
