"""
scripts/backfill_general_market_data.py
----------------------------------------
One-time 15-year historical backfill for general_market_data.

Fetches OHLCV data for all stocks in S&P 500, S&P 400, S&P 600, and
optionally ALL US-listed stocks from NASDAQ Trader, EXCLUDING any ticker
already in market_data (the Russell 2000 algorithm universe).

Features
--------
• Checkpointing   — progress saved to checkpoints/general_backfill_checkpoint.json
                    after every batch; safe to interrupt and re-run at any time.
• Deduplication   — queries BQ for existing dates before writing; never doubles data.
• Rate-limit safe — exponential back-off (60 → 120 → 240 → 480 → 960 s), then
                    graceful exit with checkpoint saved so you can resume later.
• Adaptive sleep  — inter-batch sleep increases automatically after repeated failures.
• Idempotent      — re-running the same command picks up exactly where it left off.

Usage
-----
  # S&P 500 + S&P 400 + S&P 600  (~1,500 tickers, recommended first pass)
  python -m scripts.backfill_general_market_data

  # ALL US-listed stocks (~7,000+ tickers; run after the first pass)
  python -m scripts.backfill_general_market_data --all-listed

  # Preview only — show ticker universe without touching BigQuery
  python -m scripts.backfill_general_market_data --dry-run

  # Wipe checkpoint and start fresh  (use only if you want to re-download everything)
  python -m scripts.backfill_general_market_data --reset-checkpoint

Scheduling
----------
  This script is for the ONE-TIME historical load only.
  Daily incremental updates are handled by:
      python -m scripts.update_data --update-general --prune-old

Prerequisites
-------------
  pip install yfinance google-cloud-bigquery pandas numpy pyarrow

Credentials
-----------
  Reads GOOGLE_APPLICATION_CREDENTIALS env var, or falls back to
  ../backend/secrets/gcp-service-account.json relative to the Algorithm/ dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from google.cloud import bigquery
from google.oauth2 import service_account

# ── Logging ───────────────────────────────────────────────────────────────────

_log_filename = f"backfill_general_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_filename),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

GCP_PROJECT         = "capstone-487001"
WRITE_DATASET       = "output_results"
MARKET_DATA_TABLE   = f"{GCP_PROJECT}.{WRITE_DATASET}.market_data"
GENERAL_DATA_TABLE  = f"{GCP_PROJECT}.{WRITE_DATASET}.general_market_data"
GENERAL_META_TABLE  = f"{GCP_PROJECT}.{WRITE_DATASET}.general_ticker_metadata"

BACKFILL_YEARS      = 15          # how far back to load
BATCH_SIZE          = 50          # tickers per yfinance download call
BASE_SLEEP_SEC      = 3.0         # polite pause between batches (seconds)
MAX_SLEEP_SEC       = 30.0        # upper bound on adaptive sleep
# Progressive rate-limit back-off delays (seconds)
RATE_LIMIT_BACKOFFS = [60, 120, 240, 480, 960]

_SCRIPT_DIR  = Path(__file__).resolve().parent
_ALGO_DIR    = _SCRIPT_DIR.parent
CHECKPOINT_DIR  = _ALGO_DIR / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "general_backfill_checkpoint.json"


# ── Checkpoint ────────────────────────────────────────────────────────────────

class BackfillCheckpoint:
    """
    Tracks which tickers have been fully backfilled.

    Persisted to disk as JSON after every batch so the script can resume from
    exactly where it left off after any interruption (rate limit, crash, etc.).
    """

    SCHEMA_VERSION = 2

    def __init__(self, path: Path, backfill_start: date):
        self.path           = path
        self.backfill_start = backfill_start
        self.created_at     = datetime.now().isoformat()
        self.last_saved     = datetime.now().isoformat()
        self.completed: set[str] = set()   # tickers fully done
        self.failed: dict[str, str] = {}   # ticker → error message
        self.rows_written   = 0
        self.batches_done   = 0

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "backfill_start": str(self.backfill_start),
            "created_at":     self.created_at,
            "last_saved":     datetime.now().isoformat(),
            "completed":      sorted(self.completed),
            "failed":         self.failed,
            "stats": {
                "rows_written": self.rows_written,
                "batches_done": self.batches_done,
                "tickers_completed": len(self.completed),
                "tickers_failed":    len(self.failed),
            },
        }

    @classmethod
    def from_dict(cls, d: dict, path: Path) -> "BackfillCheckpoint":
        start = date.fromisoformat(d["backfill_start"])
        cp = cls(path, start)
        cp.created_at   = d.get("created_at", datetime.now().isoformat())
        cp.completed    = set(d.get("completed", []))
        cp.failed       = d.get("failed", {})
        stats           = d.get("stats", {})
        cp.rows_written = stats.get("rows_written", 0)
        cp.batches_done = stats.get("batches_done", 0)
        return cp

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        tmp.replace(self.path)   # atomic write

    @classmethod
    def load_or_create(cls, path: Path, backfill_start: date) -> "BackfillCheckpoint":
        if path.exists():
            try:
                with open(path) as f:
                    d = json.load(f)
                cp = cls.from_dict(d, path)
                logger.info(
                    f"  Loaded checkpoint: {len(cp.completed):,} tickers done, "
                    f"{len(cp.failed):,} failed, {cp.rows_written:,} rows written"
                )
                # Warn if backfill_start changed
                if cp.backfill_start != backfill_start:
                    logger.warning(
                        f"  ⚠ Checkpoint backfill_start ({cp.backfill_start}) "
                        f"differs from current ({backfill_start}). "
                        "Using checkpoint value to stay consistent."
                    )
                return cp
            except Exception as e:
                logger.warning(f"  Could not read checkpoint ({e}); starting fresh.")
        cp = cls(path, backfill_start)
        logger.info("  No checkpoint found — starting fresh backfill.")
        return cp

    # ── State helpers ─────────────────────────────────────────────────────────

    def mark_completed(self, ticker: str, rows: int = 0) -> None:
        self.completed.add(ticker)
        self.failed.pop(ticker, None)   # clear any previous failure
        self.rows_written += rows

    def mark_failed(self, ticker: str, reason: str) -> None:
        self.failed[ticker] = reason

    def get_pending(self, all_tickers: list[str]) -> list[str]:
        done = self.completed | set(self.failed.keys())
        return [t for t in all_tickers if t not in done]

    def summary(self) -> str:
        return (
            f"Completed: {len(self.completed):,}  "
            f"Failed: {len(self.failed):,}  "
            f"Rows written: {self.rows_written:,}  "
            f"Batches done: {self.batches_done:,}"
        )


# ── Credentials ───────────────────────────────────────────────────────────────

def _build_client() -> bigquery.Client:
    creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_env and os.path.isfile(creds_env):
        logger.info(f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {creds_env}")
        creds = service_account.Credentials.from_service_account_file(creds_env)
        return bigquery.Client(project=GCP_PROJECT, credentials=creds)

    sa_path = os.path.normpath(
        os.path.join(_ALGO_DIR, "..", "backend", "secrets", "gcp-service-account.json")
    )
    if os.path.isfile(sa_path):
        logger.info(f"Using service account: {sa_path}")
        creds = service_account.Credentials.from_service_account_file(sa_path)
        return bigquery.Client(project=GCP_PROJECT, credentials=creds)

    logger.warning("Falling back to application default credentials (gcloud auth).")
    return bigquery.Client(project=GCP_PROJECT)


# ── BQ table setup ────────────────────────────────────────────────────────────

def _ensure_general_market_data_table(client: bigquery.Client) -> None:
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
        "OHLCV for S&P 500/400/600 and broad-market stocks outside the "
        "lead-lag analysis universe (market_data). "
        "NEVER read by the Algorithm pipeline."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY, field="date"
    )
    table.clustering_fields = ["ticker"]
    client.create_table(table, exists_ok=True)
    logger.info(f"  ✓ Table ready: {GENERAL_DATA_TABLE}")


def _ensure_general_ticker_metadata_table(client: bigquery.Client) -> None:
    schema = [
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
    table = bigquery.Table(GENERAL_META_TABLE, schema=schema)
    table.description = (
        "Metadata (sector, market cap, etc.) for tickers in general_market_data."
    )
    client.create_table(table, exists_ok=True)
    logger.info(f"  ✓ Table ready: {GENERAL_META_TABLE}")


# ── Ticker universe ───────────────────────────────────────────────────────────

def _fetch_sp_index_tickers() -> list[str]:
    """
    Fetch S&P 500, S&P 400, and S&P 600 constituent lists from iShares ETF
    holdings CSVs (IVV, IJH, IJR). Reuses the same approach as
    update_market_data_universe.py. No authentication required.
    """
    import requests
    from io import StringIO

    sources = [
        ("https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf"
         "/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund",
         "iShares IVV (S&P 500)"),
        ("https://www.ishares.com/us/products/239763/ishares-core-sp-midcap-etf"
         "/1467271812596.ajax?fileType=csv&fileName=IJH_holdings&dataType=fund",
         "iShares IJH (S&P 400)"),
        ("https://www.ishares.com/us/products/239775/ishares-core-sp-smallcap-etf"
         "/1467271812596.ajax?fileType=csv&fileName=IJR_holdings&dataType=fund",
         "iShares IJR (S&P 600)"),
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.ishares.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    tickers: set[str] = set()
    for url, label in sources:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()

            lines = resp.text.splitlines()
            header_idx = next(
                (i for i, line in enumerate(lines)
                 if "Ticker" in line and ("Weight" in line or "Name" in line)),
                None,
            )
            if header_idx is None:
                logger.warning(f"  {label}: could not locate header row")
                continue

            df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
            if "Ticker" not in df.columns:
                logger.warning(f"  {label}: 'Ticker' column missing")
                continue

            if "Asset Class" in df.columns:
                df = df[df["Asset Class"].astype(str).str.strip() == "Equity"]

            cleaned = (
                df["Ticker"].dropna().astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)
                .str.upper()
            )
            valid = cleaned[cleaned.str.match(r"^[A-Z]{1,5}(-[A-Z])?$")].tolist()
            before = len(tickers)
            tickers.update(valid)
            logger.info(f"  {label}: {len(valid)} tickers ({len(tickers) - before} new)")

        except Exception as e:
            logger.warning(f"  {label} fetch failed: {e}")

    return sorted(tickers)


def _fetch_nasdaq_trader_tickers() -> list[str]:
    """
    Fetch broad US market tickers from iShares ITOT (S&P Total Market ETF).
    Covers ~2,500 stocks including small/micro caps beyond S&P 1500.
    Falls back to iShares IWV (Russell 3000) if ITOT is unreachable.
    """
    import requests
    from io import StringIO

    sources = [
        ("https://www.ishares.com/us/products/239724/ishares-core-sp-total-us-stock-market-etf"
         "/1467271812596.ajax?fileType=csv&fileName=ITOT_holdings&dataType=fund",
         "iShares ITOT (S&P Total Market, ~2500 stocks)"),
        ("https://www.ishares.com/us/products/239726/ishares-russell-3000-etf"
         "/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
         "iShares IWV (Russell 3000, fallback)"),
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.ishares.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    for url, label in sources:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()

            lines = resp.text.splitlines()
            header_idx = next(
                (i for i, line in enumerate(lines)
                 if "Ticker" in line and ("Weight" in line or "Name" in line)),
                None,
            )
            if header_idx is None:
                logger.warning(f"  {label}: could not locate header row")
                continue

            df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
            if "Ticker" not in df.columns:
                logger.warning(f"  {label}: 'Ticker' column missing")
                continue

            if "Asset Class" in df.columns:
                df = df[df["Asset Class"].astype(str).str.strip() == "Equity"]

            cleaned = (
                df["Ticker"].dropna().astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)
                .str.upper()
            )
            valid = cleaned[cleaned.str.match(r"^[A-Z]{1,5}(-[A-Z])?$")].tolist()
            if valid:
                logger.info(f"  {label}: {len(valid)} tickers fetched")
                return sorted(valid)

        except Exception as e:
            logger.warning(f"  {label} fetch failed: {e}")

    logger.warning("  All broad-market sources failed — returning empty list")
    return []


def _build_candidate_universe(use_all_listed: bool) -> list[str]:
    """Build and return the full candidate ticker list before exclusion filtering."""
    logger.info("Building candidate ticker universe...")
    if use_all_listed:
        logger.info("  Mode: iShares ITOT (S&P Total Market ~2,500 stocks)")
        tickers = _fetch_nasdaq_trader_tickers()
    else:
        logger.info("  Mode: iShares IVV (S&P 500) + IJH (S&P 400) + IJR (S&P 600)")
        tickers = _fetch_sp_index_tickers()
        if not tickers:
            logger.warning("  iShares scrape returned nothing — falling back to NASDAQ Trader")
            tickers = _fetch_nasdaq_trader_tickers()
    logger.info(f"  → {len(tickers):,} candidate tickers total (before exclusion)")
    return tickers


# ── BQ helpers ────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return None


def _get_market_data_tickers(client: bigquery.Client) -> set[str]:
    """Return the set of tickers currently in market_data (exclusion set)."""
    query = f"SELECT DISTINCT ticker FROM `{MARKET_DATA_TABLE}`"
    return {row.ticker for row in client.query(query).result()}


def _get_existing_general_tickers(client: bigquery.Client) -> set[str]:
    """Return tickers already present in general_market_data."""
    try:
        query = f"SELECT DISTINCT ticker FROM `{GENERAL_DATA_TABLE}`"
        return {row.ticker for row in client.query(query).result()}
    except Exception:
        return set()


def _get_latest_dates(client: bigquery.Client) -> dict[str, date]:
    """Return {ticker: latest_date} for all tickers in general_market_data."""
    try:
        query = f"""
            SELECT ticker, MAX(DATE(date)) AS latest_date
            FROM `{GENERAL_DATA_TABLE}`
            GROUP BY ticker
        """
        df = client.query(query).to_dataframe()
        result: dict[str, date] = {}
        for _, row in df.iterrows():
            v = row.latest_date
            if isinstance(v, datetime):
                result[row.ticker] = v.date()
            elif isinstance(v, date):
                result[row.ticker] = v
            else:
                result[row.ticker] = pd.Timestamp(v).date()
        return result
    except Exception:
        return {}


# ── yfinance download ─────────────────────────────────────────────────────────

def _download_batch(
    tickers: list[str],
    start: date,
    end: date,
    rate_limit_attempt: int = 0,
) -> pd.DataFrame:
    """
    Download OHLCV for a batch of tickers via yfinance.

    On rate-limit errors, waits according to RATE_LIMIT_BACKOFFS and retries
    up to len(RATE_LIMIT_BACKOFFS) times.  Returns empty DataFrame on failure.

    Returns tidy DataFrame: ticker, date, open, high, low, close, adj_close,
                             volume, log_return
    """
    ticker_str = " ".join(tickers)
    raw        = None

    for attempt, wait_sec in enumerate(RATE_LIMIT_BACKOFFS, 1):
        try:
            raw = yf.download(
                ticker_str,
                start=str(start),
                end=str(end + timedelta(days=1)),
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            break
        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = (
                "RateLimit" in type(exc).__name__
                or "Too Many Requests" in exc_str
                or "rate limit" in exc_str.lower()
                or "429" in exc_str
            )
            if is_rate_limit and attempt < len(RATE_LIMIT_BACKOFFS):
                logger.warning(
                    f"    Rate limited (attempt {attempt}/{len(RATE_LIMIT_BACKOFFS)}). "
                    f"Waiting {wait_sec}s..."
                )
                time.sleep(wait_sec)
            else:
                logger.warning(f"    Download failed after {attempt} attempts: {exc}")
                return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # ── Normalise MultiIndex columns ──────────────────────────────────────────
    if len(tickers) == 1:
        raw.columns = [str(c).lower() for c in raw.columns]
        raw = raw.reset_index().rename(columns={"Date": "date"})
        if "adj close" in raw.columns:
            raw = raw.rename(columns={"adj close": "adj_close"})
        raw["ticker"] = tickers[0]
    else:
        raw = raw.reset_index()
        frames: list[pd.DataFrame] = []
        for tkr in tickers:
            try:
                sub = pd.DataFrame()
                sub["date"]      = raw[("Date", "")]    if ("Date", "") in raw.columns else raw["Date"]
                sub["open"]      = raw[("Open",      tkr)].values
                sub["high"]      = raw[("High",      tkr)].values
                sub["low"]       = raw[("Low",       tkr)].values
                sub["close"]     = raw[("Close",     tkr)].values
                sub["adj_close"] = raw[("Adj Close", tkr)].values
                sub["volume"]    = raw[("Volume",    tkr)].values
                sub["ticker"]    = tkr
            except KeyError:
                # Ticker not in response (no data / delisted)
                continue
            frames.append(sub)

        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)

    # ── Clean up ──────────────────────────────────────────────────────────────
    raw["date"] = pd.to_datetime(raw["date"])
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in raw.columns:
            raw[col] = float("nan")

    raw = raw.dropna(subset=["close"])
    raw["volume"] = raw["volume"].fillna(0).astype("int64")

    # Compute log_return per ticker (sorted within each group)
    raw = raw.sort_values(["ticker", "date"])
    raw["log_return"] = (
        raw.groupby("ticker")["close"]
        .transform(lambda s: np.log(s / s.shift(1)))
    )

    return raw[["ticker", "date", "open", "high", "low", "close",
                "adj_close", "volume", "log_return"]]


# ── BQ upload ─────────────────────────────────────────────────────────────────

_OHLCV_SCHEMA = [
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


def _upload_df(client: bigquery.Client, df: pd.DataFrame, dry_run: bool) -> int:
    if df.empty:
        return 0
    if dry_run:
        logger.info(f"    [DRY RUN] Would write {len(df):,} rows")
        return 0
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=False,
        schema=_OHLCV_SCHEMA,
    )
    job = client.load_table_from_dataframe(df, GENERAL_DATA_TABLE, job_config=job_config)
    job.result()
    return len(df)


# ── Metadata fetch ────────────────────────────────────────────────────────────

def _fetch_and_upload_metadata(
    client: bigquery.Client,
    tickers: list[str],
    dry_run: bool,
) -> int:
    """
    Fetch yfinance .info for every ticker and MERGE into general_ticker_metadata.
    Returns the number of tickers processed.
    """
    logger.info(f"\nFetching metadata for {len(tickers):,} tickers...")
    now = datetime.now()
    records = []

    for i, tkr in enumerate(tickers, 1):
        if i % 100 == 0:
            logger.info(f"  Metadata progress: {i:,} / {len(tickers):,}")
            time.sleep(1)   # brief pause every 100 calls
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
            logger.debug(f"  Metadata fetch failed for {tkr}: {e}")
            records.append({
                "ticker":       tkr,
                "company_name": tkr,
                "sector":       None, "industry":    None,
                "market_cap":   None, "pe_ratio":    None,
                "country":      None, "exchange":    None,
                "currency":     None, "website":     None,
                "description":  None, "employees":   None,
                "last_updated": now,  "is_active":   False,
            })

    df = pd.DataFrame(records)
    df["market_cap"] = df["market_cap"].astype("Int64")
    df["employees"]  = df["employees"].astype("Int64")

    if dry_run:
        logger.info(f"  [DRY RUN] Would upsert {len(df):,} rows into {GENERAL_META_TABLE}")
        return len(df)

    _ensure_general_ticker_metadata_table(client)

    # Write to staging → MERGE
    staging_id = (
        f"{GCP_PROJECT}.{WRITE_DATASET}."
        f"_gen_meta_staging_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
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
    staging_table = bigquery.Table(staging_id, schema=meta_schema)
    client.create_table(staging_table, exists_ok=True)
    job = client.load_table_from_dataframe(
        df, staging_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE", schema=meta_schema
        ),
    )
    job.result()

    merge_sql = f"""
        MERGE `{GENERAL_META_TABLE}` T
        USING `{staging_id}` S ON T.ticker = S.ticker
        WHEN MATCHED THEN UPDATE SET
            T.company_name = S.company_name, T.sector = S.sector,
            T.industry = S.industry,         T.market_cap = S.market_cap,
            T.pe_ratio = S.pe_ratio,         T.country = S.country,
            T.exchange = S.exchange,         T.currency = S.currency,
            T.website = S.website,           T.description = S.description,
            T.employees = S.employees,       T.last_updated = S.last_updated,
            T.is_active = S.is_active
        WHEN NOT MATCHED THEN INSERT
            (ticker, company_name, sector, industry, market_cap, pe_ratio,
             country, exchange, currency, website, description, employees,
             last_updated, is_active)
            VALUES
            (S.ticker, S.company_name, S.sector, S.industry, S.market_cap,
             S.pe_ratio, S.country, S.exchange, S.currency, S.website,
             S.description, S.employees, S.last_updated, S.is_active)
    """
    client.query(merge_sql).result()

    try:
        client.delete_table(staging_id)
    except Exception:
        pass

    logger.info(f"  ✓ Metadata upserted: {len(df):,} tickers")
    return len(df)


# ── Core backfill loop ────────────────────────────────────────────────────────

def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def run_backfill(
    client: bigquery.Client,
    candidates: list[str],
    checkpoint: BackfillCheckpoint,
    dry_run: bool,
    fetch_metadata: bool,
) -> None:
    """
    Main backfill loop.

    For each batch of BATCH_SIZE pending tickers:
      1. Determine start date (backfill_start or latest BQ date + 1 day)
      2. Download OHLCV from yfinance
      3. Filter out rows already in BQ
      4. Upload to general_market_data
      5. Mark tickers completed in checkpoint + save checkpoint
      6. Sleep (adaptive)
    """
    today         = date.today()
    backfill_start = checkpoint.backfill_start

    # Get pending tickers (remove already-completed and failed from checkpoint)
    pending = checkpoint.get_pending(candidates)
    if not pending:
        logger.info("  ✓ All tickers already completed according to checkpoint.")
        return

    total_batches = math.ceil(len(pending) / BATCH_SIZE)
    logger.info(
        f"\n{'=' * 60}\n"
        f"  Pending tickers: {len(pending):,}  |  "
        f"Batches: {total_batches:,}  |  "
        f"Batch size: {BATCH_SIZE}\n"
        f"  Backfill window: {backfill_start} → {today}\n"
        f"{'=' * 60}"
    )

    if not dry_run:
        _ensure_general_market_data_table(client)

    # Load latest dates from BQ once at start (re-query is expensive)
    logger.info("  Loading existing BQ coverage (latest date per ticker)...")
    latest_dates = _get_latest_dates(client) if not dry_run else {}
    logger.info(f"  BQ coverage loaded for {len(latest_dates):,} existing tickers.")

    # Adaptive sleep: increases when we hit rate limits
    current_sleep = BASE_SLEEP_SEC
    consecutive_failures = 0

    for batch_num, batch in enumerate(_batched(pending, BATCH_SIZE), 1):
        logger.info(
            f"\n  Batch {batch_num}/{total_batches} "
            f"({len(batch)} tickers — first: {batch[0]}, last: {batch[-1]})"
        )

        # Determine per-ticker start dates; use earliest for one batch download
        per_ticker_start: dict[str, date] = {}
        for tkr in batch:
            last_in_bq = latest_dates.get(tkr)
            if last_in_bq:
                per_ticker_start[tkr] = last_in_bq + timedelta(days=1)
            else:
                per_ticker_start[tkr] = backfill_start

        earliest_start = min(per_ticker_start.values())
        if earliest_start > today:
            logger.info(f"    All tickers in batch already up to date — skipping.")
            for tkr in batch:
                checkpoint.mark_completed(tkr, rows=0)
            checkpoint.batches_done += 1
            checkpoint.save()
            continue

        # ── Download ──────────────────────────────────────────────────────────
        df = _download_batch(batch, earliest_start, today)

        if df.empty:
            consecutive_failures += 1
            logger.warning(
                f"    No data returned for batch {batch_num} "
                f"(consecutive failures: {consecutive_failures})"
            )
            # Mark all tickers in this batch as failed temporarily
            for tkr in batch:
                checkpoint.mark_failed(tkr, "No data returned from yfinance")
            checkpoint.save()

            if consecutive_failures >= len(RATE_LIMIT_BACKOFFS):
                logger.error(
                    f"\n  ⛔ {consecutive_failures} consecutive empty batches. "
                    f"Checkpoint saved at {checkpoint.path}. "
                    "Re-run this script to resume."
                )
                break

            # Increase sleep before next attempt
            current_sleep = min(current_sleep * 2, MAX_SLEEP_SEC)
            time.sleep(current_sleep)
            continue

        consecutive_failures = 0  # reset on success

        # ── Per-ticker dedup + filter ──────────────────────────────────────────
        rows_to_write: list[pd.DataFrame] = []
        ticker_row_counts: dict[str, int] = {}

        for tkr, grp in df.groupby("ticker"):
            tkr = str(tkr)
            cutoff = latest_dates.get(tkr, date.min)
            new_rows = grp[grp["date"].dt.date > cutoff].copy()
            if not new_rows.empty:
                rows_to_write.append(new_rows)
                ticker_row_counts[tkr] = len(new_rows)

        # Mark tickers with no new rows as completed (already up to date)
        no_new_rows_tickers = [t for t in batch if t not in ticker_row_counts
                                and t in df["ticker"].unique()]
        for tkr in no_new_rows_tickers:
            checkpoint.mark_completed(tkr, rows=0)

        if rows_to_write:
            upload_df = pd.concat(rows_to_write, ignore_index=True)
            rows_written = _upload_df(client, upload_df, dry_run)

            # Update in-memory latest_dates cache so subsequent batches are accurate
            for tkr, grp in upload_df.groupby("ticker"):
                latest_dates[str(tkr)] = grp["date"].max().date()

            # Mark tickers in this upload as completed
            for tkr, n_rows in ticker_row_counts.items():
                checkpoint.mark_completed(tkr, rows=n_rows)

            if not dry_run:
                logger.info(
                    f"    ✓ Wrote {rows_written:,} rows for "
                    f"{len(ticker_row_counts)} tickers"
                )
        else:
            logger.info(f"    No new rows after deduplication for batch {batch_num}")

        # Mark any tickers in batch not returned by yfinance as failed
        returned_tickers = set(df["ticker"].unique())
        for tkr in batch:
            if tkr not in returned_tickers and tkr not in checkpoint.completed:
                checkpoint.mark_failed(tkr, "Not returned by yfinance (possibly delisted)")

        checkpoint.batches_done += 1
        checkpoint.save()

        # ── Progress report ───────────────────────────────────────────────────
        if batch_num % 10 == 0 or batch_num == total_batches:
            pct = 100 * (checkpoint.batches_done) / total_batches
            logger.info(
                f"\n  ── Progress: {pct:.1f}% ──  {checkpoint.summary()}\n"
            )

        # ── Adaptive sleep ────────────────────────────────────────────────────
        time.sleep(current_sleep)
        # Slowly recover sleep toward BASE after a successful batch
        if current_sleep > BASE_SLEEP_SEC:
            current_sleep = max(BASE_SLEEP_SEC, current_sleep * 0.9)

    logger.info(f"\n  Final checkpoint: {checkpoint.summary()}")

    # ── Metadata (optional) ───────────────────────────────────────────────────
    if fetch_metadata and not dry_run:
        all_done = sorted(checkpoint.completed)
        _fetch_and_upload_metadata(client, all_done, dry_run)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="15-year historical backfill for general_market_data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # S&P 500 + 400 + 600 (~1,500 tickers — start here)
  python -m scripts.backfill_general_market_data

  # All US-listed stocks (~7,000+ tickers — run after the SP pass)
  python -m scripts.backfill_general_market_data --all-listed

  # Preview only (no BQ writes)
  python -m scripts.backfill_general_market_data --dry-run

  # Reset checkpoint (re-downloads everything from scratch)
  python -m scripts.backfill_general_market_data --reset-checkpoint
""",
    )
    parser.add_argument(
        "--all-listed", action="store_true",
        help="Use NASDAQ Trader (~7,000 tickers) instead of iShares S&P indices (~1,500).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be downloaded/written without touching BigQuery.",
    )
    parser.add_argument(
        "--reset-checkpoint", action="store_true",
        help="Delete existing checkpoint and start fresh (re-downloads everything).",
    )
    parser.add_argument(
        "--fetch-metadata", action="store_true",
        help=(
            "After OHLCV backfill, also fetch yfinance .info and populate "
            "general_ticker_metadata. Adds significant extra API call time."
        ),
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GENERAL MARKET DATA — HISTORICAL BACKFILL")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        logger.info("MODE: DRY RUN — no writes will be made")
    logger.info("=" * 60)

    # ── Checkpoint setup ──────────────────────────────────────────────────────
    if args.reset_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info(f"  Checkpoint deleted: {CHECKPOINT_FILE}")

    backfill_start = date.today() - timedelta(days=BACKFILL_YEARS * 365)
    checkpoint = BackfillCheckpoint.load_or_create(CHECKPOINT_FILE, backfill_start)

    # ── BQ client ─────────────────────────────────────────────────────────────
    client = _build_client()

    # ── Build ticker universe ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Building ticker universe")
    logger.info("=" * 60)
    candidates_raw = _build_candidate_universe(args.all_listed)

    logger.info("\nLoading market_data exclusion set from BigQuery...")
    market_data_tickers = _get_market_data_tickers(client)
    logger.info(f"  {len(market_data_tickers):,} tickers in market_data (excluded)")

    candidates = [t for t in candidates_raw if t not in market_data_tickers]
    excluded   = len(candidates_raw) - len(candidates)
    logger.info(
        f"  {len(candidates_raw):,} raw candidates  "
        f"- {excluded:,} market_data overlaps  "
        f"= {len(candidates):,} for general_market_data"
    )

    if args.dry_run:
        pending = checkpoint.get_pending(candidates)
        logger.info(f"\n[DRY RUN] Pending: {len(pending):,} tickers to download")
        logger.info(f"  Checkpoint completed: {len(checkpoint.completed):,}")
        logger.info(f"  Checkpoint failed:    {len(checkpoint.failed):,}")
        if pending:
            logger.info(f"  Sample pending: {pending[:20]}")
        logger.info("\nDry run complete — no BigQuery writes made.")
        return

    # ── Run backfill ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Running OHLCV backfill")
    logger.info("=" * 60)
    run_backfill(
        client     = client,
        candidates = candidates,
        checkpoint = checkpoint,
        dry_run    = args.dry_run,
        fetch_metadata = args.fetch_metadata,
    )

    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {_log_filename}")
    logger.info(f"Checkpoint: {CHECKPOINT_FILE}")
    logger.info(
        f"Failed tickers ({len(checkpoint.failed)}): "
        + (str(sorted(checkpoint.failed.keys())[:20]) if checkpoint.failed else "none")
    )
    logger.info("=" * 60)
    logger.info(
        "\nNext step: run daily incremental updates with:\n"
        "  python -m scripts.update_data --update-general --prune-old"
    )


if __name__ == "__main__":
    main()
