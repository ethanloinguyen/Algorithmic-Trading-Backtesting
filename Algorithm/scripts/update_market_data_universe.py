"""
scripts/update_market_data_universe.py
---------------------------------------
Fix the Russell 3000 top-2000 gap in market_data.

Background
----------
market_data currently holds 1,683 tickers instead of the expected ~2,000 because
the original seed list was a stale snapshot of the target universe.

The intended market_data universe is the TOP 2,000 stocks from the Russell 3000
by market cap (i.e. ranks 1–2,000).  This is NOT the Russell 2000 (which is
ranks 1,001–3,000 — the small-cap tail).  Concretely:

    Russell 3000 ranks 1–1,000   →  Russell 1000 (large-cap)   — ALL included
    Russell 3000 ranks 1,001–2,000  →  top half of Russell 2000  — ALL included
    Russell 3000 ranks 2,001–3,000  →  bottom half of Russell 2000 — EXCLUDED

This script:
  1. Fetches the current Russell 3000 constituent list from the iShares IWV ETF
     holdings CSV (most accurate free source; updated daily by iShares).
  2. Sorts holdings by weight (% of ETF) descending — weight is proportional to
     market cap, so rank 1 is the largest stock.
  3. Takes the top 2,000 holdings by weight → this is the target universe.
  4. Falls back to IWB (Russell 1000) + top-1,000 by weight from IWM (Russell 2000)
     if IWV is unreachable.
  5. Identifies which tickers are missing from market_data.
  6. Downloads 15 years of OHLCV for the missing tickers via yfinance.
  7. Appends those rows to market_data (same table the algorithm pipeline uses).
  8. Checkpoints progress so the script is safe to interrupt and re-run.

Run ONCE after backfill_general_market_data.py is complete.  After that, daily
updates via update_data.py keep market_data current.

IMPORTANT
---------
  This script writes to market_data — the live algorithm pipeline table.
  Always run with --dry-run first to confirm tickers and row counts look right.

Usage
-----
  # Dry run (shows what would be added, no BQ writes)
  python -m scripts.update_market_data_universe --dry-run

  # Live run (appends missing tickers to market_data)
  python -m scripts.update_market_data_universe

  # Reset checkpoint and restart from scratch
  python -m scripts.update_market_data_universe --reset-checkpoint

  # Override the target universe size (default: 2000)
  python -m scripts.update_market_data_universe --target-count 2000

Prerequisites
-------------
  pip install yfinance google-cloud-bigquery pandas numpy pyarrow requests
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

_log_filename = f"update_universe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

GCP_PROJECT       = "capstone-487001"
WRITE_DATASET     = "output_results"
MARKET_DATA_TABLE = f"{GCP_PROJECT}.{WRITE_DATASET}.market_data"

BACKFILL_YEARS = 15
BATCH_SIZE     = 50
BASE_SLEEP_SEC = 3.0
RATE_LIMIT_BACKOFFS = [60, 120, 240, 480, 960]

_SCRIPT_DIR     = Path(__file__).resolve().parent
_ALGO_DIR       = _SCRIPT_DIR.parent
CHECKPOINT_DIR  = _ALGO_DIR / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "universe_fix_checkpoint.json"

# iShares ETF holdings CSV URLs
# IWV = Russell 3000 ETF  (~3,000 holdings; take top 2,000 by weight)
# IWB = Russell 1000 ETF  (~1,000 holdings; all included)
# IWM = Russell 2000 ETF  (~2,000 holdings; take top 1,000 by weight as fallback)
_ISHARES_IWV_URL = (
    "https://www.ishares.com/us/products/239726/ishares-russell-3000-etf"
    "/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
)
_ISHARES_IWB_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf"
    "/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)
_ISHARES_IWM_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
    "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)
_ISHARES_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.ishares.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ── Checkpoint (reuse pattern from backfill script) ───────────────────────────

class UniverseFixCheckpoint:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self.path           = path
        self.created_at     = datetime.now().isoformat()
        self.completed: set[str] = set()
        self.failed: dict[str, str] = {}
        self.rows_written   = 0
        self.batches_done   = 0

    def to_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "created_at":     self.created_at,
            "last_saved":     datetime.now().isoformat(),
            "completed":      sorted(self.completed),
            "failed":         self.failed,
            "stats": {
                "rows_written": self.rows_written,
                "batches_done": self.batches_done,
            },
        }

    @classmethod
    def from_dict(cls, d: dict, path: Path) -> "UniverseFixCheckpoint":
        cp = cls(path)
        cp.created_at   = d.get("created_at", datetime.now().isoformat())
        cp.completed    = set(d.get("completed", []))
        cp.failed       = d.get("failed", {})
        stats           = d.get("stats", {})
        cp.rows_written = stats.get("rows_written", 0)
        cp.batches_done = stats.get("batches_done", 0)
        return cp

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        tmp.replace(self.path)

    @classmethod
    def load_or_create(cls, path: Path) -> "UniverseFixCheckpoint":
        if path.exists():
            try:
                with open(path) as f:
                    d = json.load(f)
                cp = cls.from_dict(d, path)
                logger.info(
                    f"  Loaded checkpoint: {len(cp.completed):,} tickers done, "
                    f"{len(cp.failed):,} failed"
                )
                return cp
            except Exception as e:
                logger.warning(f"  Could not read checkpoint ({e}); starting fresh.")
        cp = cls(path)
        logger.info("  No checkpoint found — starting fresh.")
        return cp

    def mark_completed(self, ticker: str, rows: int = 0) -> None:
        self.completed.add(ticker)
        self.failed.pop(ticker, None)
        self.rows_written += rows

    def mark_failed(self, ticker: str, reason: str) -> None:
        self.failed[ticker] = reason

    def get_pending(self, all_tickers: list[str]) -> list[str]:
        done = self.completed | set(self.failed.keys())
        return [t for t in all_tickers if t not in done]


# ── Credentials ───────────────────────────────────────────────────────────────

def _build_client() -> bigquery.Client:
    creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_env and os.path.isfile(creds_env):
        creds = service_account.Credentials.from_service_account_file(creds_env)
        return bigquery.Client(project=GCP_PROJECT, credentials=creds)
    sa_path = os.path.normpath(
        os.path.join(_ALGO_DIR, "..", "backend", "secrets", "gcp-service-account.json")
    )
    if os.path.isfile(sa_path):
        creds = service_account.Credentials.from_service_account_file(sa_path)
        return bigquery.Client(project=GCP_PROJECT, credentials=creds)
    return bigquery.Client(project=GCP_PROJECT)


# ── Russell 3000 top-2000 constituent fetching ────────────────────────────────

def _parse_ishares_csv(url: str, label: str) -> pd.DataFrame:
    """
    Download an iShares ETF holdings CSV and return a cleaned DataFrame with
    columns: ticker, weight_pct.

    iShares CSV layout:
      • First N lines: fund-level metadata (we skip until we find the header row)
      • Header row: contains 'Ticker', 'Name', 'Asset Class', 'Weight (%)', etc.
      • Data rows: one per holding

    Returns an empty DataFrame on any error.
    """
    try:
        import requests
        from io import StringIO

        resp = requests.get(url, headers=_ISHARES_HEADERS, timeout=30)
        resp.raise_for_status()

        lines = resp.text.splitlines()

        # Find the header row — it contains both 'Ticker' and 'Weight'
        header_idx = None
        for i, line in enumerate(lines):
            if "Ticker" in line and ("Weight" in line or "Name" in line):
                header_idx = i
                break

        if header_idx is None:
            logger.warning(f"  {label}: could not locate header row in CSV")
            return pd.DataFrame()

        csv_content = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(csv_content))

        if "Ticker" not in df.columns:
            logger.warning(f"  {label}: 'Ticker' column missing")
            return pd.DataFrame()

        # Keep only equity rows
        if "Asset Class" in df.columns:
            df = df[df["Asset Class"].astype(str).str.strip() == "Equity"]

        # Normalise ticker symbols: BRK.B → BRK-B, drop warrants/units
        df["ticker"] = (
            df["Ticker"].dropna().astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .str.upper()
        )
        valid_mask = df["ticker"].str.match(r"^[A-Z]{1,5}(-[A-Z])?$")
        df = df[valid_mask].copy()

        # Parse weight column (may be named "Weight (%)" or "Weightings")
        weight_col = next(
            (c for c in df.columns if "weight" in c.lower()), None
        )
        if weight_col:
            df["weight_pct"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
        else:
            # No weight column: assign equal weight so sorting is still possible
            df["weight_pct"] = 1.0

        result = df[["ticker", "weight_pct"]].drop_duplicates("ticker").reset_index(drop=True)
        logger.info(f"  ✓ {label}: {len(result)} equity holdings parsed")
        return result

    except Exception as e:
        logger.warning(f"  {label} fetch failed: {e}")
        return pd.DataFrame()


def _fetch_iwv_top2000(target: int = 2000) -> list[str]:
    """
    Primary source: iShares IWV (Russell 3000 ETF).

    Downloads all ~3,000 holdings, sorts by weight (%) descending
    (weight is proportional to market cap), and returns the top `target` tickers.
    This correctly captures ranks 1–2,000 of the Russell 3000.
    """
    df = _parse_ishares_csv(_ISHARES_IWV_URL, "iShares IWV (Russell 3000)")
    if df.empty:
        return []

    df = df.sort_values("weight_pct", ascending=False).head(target)
    tickers = df["ticker"].tolist()
    logger.info(
        f"  IWV top-{target} by weight: {len(tickers)} tickers "
        f"(weight range: {df['weight_pct'].max():.4f}% – {df['weight_pct'].min():.4f}%)"
    )
    return tickers


def _fetch_iwb_plus_iwm_top1000() -> list[str]:
    """
    Fallback source: IWB (Russell 1000, all) + top-1,000 of IWM by weight.

    IWB covers ranks 1–1,000 exactly.
    IWM covers ranks 1,001–3,000; taking the top 1,000 by weight gives
    ranks 1,001–2,000 (the mid-cap portion we want).

    Combined → the same top-2,000 universe as IWV, via two separate ETFs.
    """
    df_iwb = _parse_ishares_csv(_ISHARES_IWB_URL, "iShares IWB (Russell 1000)")
    df_iwm = _parse_ishares_csv(_ISHARES_IWM_URL, "iShares IWM (Russell 2000)")

    tickers: set[str] = set()

    if not df_iwb.empty:
        # Russell 1000 → include all
        tickers.update(df_iwb["ticker"].tolist())
        logger.info(f"  IWB: all {len(df_iwb)} Russell 1000 tickers included")

    if not df_iwm.empty:
        # Russell 2000 → take top 1,000 by weight (= ranks 1,001–2,000)
        top_iwm = (
            df_iwm.sort_values("weight_pct", ascending=False)
            .head(1000)["ticker"]
            .tolist()
        )
        tickers.update(top_iwm)
        logger.info(f"  IWM top-1,000 by weight: {len(top_iwm)} tickers included")

    return sorted(tickers)


def fetch_russell3000_top2000_tickers(target: int = 2000) -> list[str]:
    """
    Return the best available list of the top `target` stocks from the
    Russell 3000 by market cap (i.e. ranks 1 through `target`).

    Strategy (in order of preference):
      1. iShares IWV (Russell 3000 ETF) — sort by weight, take top 2,000
         Most accurate: one ETF, one sort, exact result.
      2. iShares IWB (Russell 1000, all) + top-1,000 of IWM by weight
         Reliable fallback using two ETFs.
      3. Empty list — caller will abort with a clear error.
    """
    logger.info(f"Fetching top-{target} Russell 3000 constituents...")

    # ── Strategy 1: IWV ───────────────────────────────────────────────────
    tickers = _fetch_iwv_top2000(target)
    if len(tickers) >= int(target * 0.90):   # accept if ≥ 90% of target
        logger.info(f"  Using IWV: {len(tickers)} tickers (target {target})")
        return sorted(tickers)

    logger.warning(
        f"  IWV returned only {len(tickers)} tickers — "
        "trying IWB + IWM fallback..."
    )

    # ── Strategy 2: IWB + IWM top-1000 ───────────────────────────────────
    tickers = _fetch_iwb_plus_iwm_top1000()
    if len(tickers) >= int(target * 0.90):
        logger.info(f"  Using IWB + IWM fallback: {len(tickers)} tickers")
        return sorted(tickers)

    logger.error(
        f"  Both iShares sources failed or returned too few tickers ({len(tickers)}). "
        "Check network connectivity and retry."
    )
    return sorted(tickers)   # return what we have; main() will warn


# ── BQ helpers ────────────────────────────────────────────────────────────────

def _get_existing_market_data_tickers(client: bigquery.Client) -> set[str]:
    query = f"SELECT DISTINCT ticker FROM `{MARKET_DATA_TABLE}`"
    return {row.ticker for row in client.query(query).result()}


def _get_latest_dates_market_data(client: bigquery.Client) -> dict[str, date]:
    query = f"""
        SELECT ticker, MAX(DATE(date)) AS latest_date
        FROM `{MARKET_DATA_TABLE}`
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


# ── yfinance download (same logic as backfill script) ────────────────────────

def _download_batch(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
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
                or "429" in exc_str
            )
            if is_rate_limit and attempt < len(RATE_LIMIT_BACKOFFS):
                logger.warning(f"    Rate limited (attempt {attempt}). Waiting {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                logger.warning(f"    Download failed: {exc}")
                return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

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
                continue
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)

    raw["date"] = pd.to_datetime(raw["date"])
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in raw.columns:
            raw[col] = float("nan")
    raw = raw.dropna(subset=["close"])
    raw["volume"] = raw["volume"].fillna(0).astype("int64")
    raw = raw.sort_values(["ticker", "date"])
    raw["log_return"] = (
        raw.groupby("ticker")["close"]
        .transform(lambda s: np.log(s / s.shift(1)))
    )
    return raw[["ticker", "date", "open", "high", "low", "close",
                "adj_close", "volume", "log_return"]]


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
        logger.info(f"    [DRY RUN] Would write {len(df):,} rows to market_data")
        return 0
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=False,
        schema=_OHLCV_SCHEMA,
    )
    job = client.load_table_from_dataframe(df, MARKET_DATA_TABLE, job_config=job_config)
    job.result()
    return len(df)


# ── Core fill loop ────────────────────────────────────────────────────────────

def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def run_universe_fill(
    client: bigquery.Client,
    missing_tickers: list[str],
    checkpoint: UniverseFixCheckpoint,
    dry_run: bool,
) -> None:
    today          = date.today()
    backfill_start = today - timedelta(days=BACKFILL_YEARS * 365)

    pending = checkpoint.get_pending(missing_tickers)
    if not pending:
        logger.info("  ✓ All missing tickers already backfilled (checkpoint complete).")
        return

    total_batches = math.ceil(len(pending) / BATCH_SIZE)
    logger.info(
        f"\n  Missing tickers to backfill: {len(pending):,}  |  "
        f"Batches: {total_batches:,}\n"
        f"  Window: {backfill_start} → {today}"
    )

    logger.info("  Loading existing BQ coverage for market_data...")
    latest_dates = _get_latest_dates_market_data(client) if not dry_run else {}
    logger.info(f"  BQ coverage loaded for {len(latest_dates):,} tickers.")

    current_sleep       = BASE_SLEEP_SEC
    consecutive_failures = 0

    for batch_num, batch in enumerate(_batched(pending, BATCH_SIZE), 1):
        logger.info(
            f"\n  Batch {batch_num}/{total_batches} "
            f"({len(batch)} tickers — {batch[0]} ... {batch[-1]})"
        )

        per_ticker_start: dict[str, date] = {
            tkr: (latest_dates.get(tkr, backfill_start - timedelta(days=1)) + timedelta(days=1))
            for tkr in batch
        }
        earliest_start = min(per_ticker_start.values())

        if earliest_start > today:
            logger.info(f"    All tickers already up to date — skipping.")
            for tkr in batch:
                checkpoint.mark_completed(tkr, rows=0)
            checkpoint.save()
            continue

        df = _download_batch(batch, earliest_start, today)

        if df.empty:
            consecutive_failures += 1
            logger.warning(
                f"    Empty response (consecutive failures: {consecutive_failures})"
            )
            for tkr in batch:
                checkpoint.mark_failed(tkr, "No data returned")
            checkpoint.save()

            if consecutive_failures >= len(RATE_LIMIT_BACKOFFS):
                logger.error(
                    f"\n  ⛔ {consecutive_failures} consecutive empty batches. "
                    "Checkpoint saved. Re-run this script to resume."
                )
                break

            current_sleep = min(current_sleep * 2, 30.0)
            time.sleep(current_sleep)
            continue

        consecutive_failures = 0

        rows_to_write: list[pd.DataFrame] = []
        ticker_row_counts: dict[str, int] = {}

        for tkr, grp in df.groupby("ticker"):
            tkr = str(tkr)
            cutoff = latest_dates.get(tkr, date.min)
            new_rows = grp[grp["date"].dt.date > cutoff].copy()
            if not new_rows.empty:
                rows_to_write.append(new_rows)
                ticker_row_counts[tkr] = len(new_rows)

        if rows_to_write:
            upload_df = pd.concat(rows_to_write, ignore_index=True)
            rows_written = _upload_df(client, upload_df, dry_run)

            for tkr, grp in upload_df.groupby("ticker"):
                latest_dates[str(tkr)] = grp["date"].max().date()
            for tkr, n_rows in ticker_row_counts.items():
                checkpoint.mark_completed(tkr, rows=n_rows)
            if not dry_run:
                logger.info(
                    f"    ✓ Wrote {rows_written:,} rows for "
                    f"{len(ticker_row_counts)} tickers"
                )
        else:
            logger.info(f"    No new rows after dedup for batch {batch_num}")

        returned_tickers = set(df["ticker"].unique())
        for tkr in batch:
            if tkr not in returned_tickers and tkr not in checkpoint.completed:
                checkpoint.mark_failed(tkr, "Not returned by yfinance")

        checkpoint.batches_done += 1
        checkpoint.save()

        if batch_num % 5 == 0 or batch_num == total_batches:
            pct = 100 * batch_num / total_batches
            logger.info(
                f"\n  ── Progress: {pct:.1f}%  "
                f"done={len(checkpoint.completed):,}  "
                f"rows={checkpoint.rows_written:,}\n"
            )

        time.sleep(current_sleep)
        if current_sleep > BASE_SLEEP_SEC:
            current_sleep = max(BASE_SLEEP_SEC, current_sleep * 0.9)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix Russell 2000 gap: add missing tickers to market_data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Preview what's missing (no writes)
  python -m scripts.update_market_data_universe --dry-run

  # Live run
  python -m scripts.update_market_data_universe

  # Reset checkpoint and restart
  python -m scripts.update_market_data_universe --reset-checkpoint
""",
    )
    parser.add_argument("--dry-run",          action="store_true")
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument(
        "--target-count", type=int, default=2000,
        help="Expected size of the Russell 2000 (default: 2000).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MARKET DATA UNIVERSE FIX — Russell 3000 top-2,000 gap")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        logger.info("MODE: DRY RUN")
    logger.info("=" * 60)

    if args.reset_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info(f"  Checkpoint deleted: {CHECKPOINT_FILE}")

    client     = _build_client()
    checkpoint = UniverseFixCheckpoint.load_or_create(CHECKPOINT_FILE)

    # ── Step 1: Fetch top-2000 of Russell 3000 ───────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Fetching top-2,000 Russell 3000 constituents (IWV)")
    logger.info("=" * 60)
    target_tickers = fetch_russell3000_top2000_tickers(target=args.target_count)
    logger.info(f"  Target universe: {len(target_tickers)} tickers fetched")

    if not target_tickers:
        logger.error("  Could not retrieve target universe. Check network and retry.")
        sys.exit(1)

    # Sanity check: if we got far fewer than expected, something went wrong
    if len(target_tickers) < int(args.target_count * 0.80):
        logger.error(
            f"  ⛔ Only {len(target_tickers)} tickers returned "
            f"(expected ~{args.target_count}). "
            "iShares may be rate-limiting or the CSV format changed. "
            "Re-run or check the URL manually."
        )
        sys.exit(1)

    # ── Step 2: Compare with existing market_data ─────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Comparing with existing market_data")
    logger.info("=" * 60)
    existing = _get_existing_market_data_tickers(client)
    logger.info(f"  market_data currently has: {len(existing):,} tickers")
    logger.info(f"  Target universe:            {len(target_tickers):,} tickers")

    missing  = sorted(t for t in target_tickers if t not in existing)
    extra    = sorted(t for t in existing if t not in set(target_tickers))
    logger.info(f"  Missing from market_data:   {len(missing)} tickers  ← will be added")
    logger.info(f"  In market_data but not in target universe: {len(extra)} tickers  ← left as-is")

    if not missing:
        logger.info("  ✓ market_data already contains all target universe tickers!")
        return

    logger.info(f"\n  Sample of missing tickers (first 30): {missing[:30]}")

    if args.dry_run:
        logger.info(
            f"\n[DRY RUN] Would backfill {len(missing)} tickers into market_data.\n"
            f"  Estimated new rows: ~{len(missing) * 252 * BACKFILL_YEARS:,}\n"
            f"  Dry run complete — no writes made."
        )
        return

    # ── Step 3: Backfill missing tickers ─────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Backfilling missing tickers into market_data")
    logger.info("=" * 60)
    run_universe_fill(client, missing, checkpoint, dry_run=False)

    logger.info("\n" + "=" * 60)
    logger.info("UNIVERSE FIX COMPLETE")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Tickers added:  {len(checkpoint.completed):,}")
    logger.info(f"  Rows written:   {checkpoint.rows_written:,}")
    logger.info(f"  Failed:         {len(checkpoint.failed):,}")
    if checkpoint.failed:
        logger.info(f"  Failed tickers: {sorted(checkpoint.failed.keys())}")
    logger.info("=" * 60)
    logger.info(
        "\nNext: re-run the universe quality filter to pick up the new tickers:\n"
        "  python run_local.py --mode setup --skip-historical"
    )


if __name__ == "__main__":
    main()
