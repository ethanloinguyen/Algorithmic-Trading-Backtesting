# Lead-Lag Pipeline — Local Runner Guide

This package lets you run the full pipeline locally while writing all results to BigQuery on GCP. The Cloud Run jobs, Cloud Workflows, and Cloud Scheduler are replaced by a single Python script (`run_local.py`) that orchestrates the same steps sequentially (or with local multiprocessing for the pair computation step).

---

## Architecture: Cloud vs Local

| Component | Cloud Version | Local Version |
|---|---|---|
| Orchestration | Cloud Workflows | `run_local.py` (Python) |
| Residuals job | Cloud Run Job | `run_step_residuals()` (in-process) |
| Pair jobs (×50) | 50 parallel Cloud Run Jobs | `ProcessPoolExecutor` (local CPUs) |
| Aggregation job | Cloud Run Job | `run_step_aggregation()` (in-process) |
| Scheduling | Cloud Scheduler | Cron / run manually |
| **Output (BigQuery)** | **Same** | **Same — identical BQ tables** |

The BigQuery output is 100% identical. Once the local pipeline is run, the FastAPI backend and frontend will work exactly as they would with Cloud Run output.

---

## Prerequisites

### 1. Python 3.9+
```bash
python3 --version
```

### 2. Google Cloud SDK
```bash
# Install: https://cloud.google.com/sdk/docs/install
gcloud --version
```

### 3. Authenticate with GCP
```bash
# This stores credentials locally for BigQuery access
gcloud auth application-default login

# Set your project
gcloud config set project capstone-487001
```

You need these IAM roles on the service account or your personal account:
- `roles/bigquery.dataEditor`
- `roles/bigquery.jobUser`
- `roles/storage.objectAdmin` (for FF factor downloads cached to GCS)

---

## Installation

```bash
cd lead_lag_local/
bash install.sh
source .venv/bin/activate
```

This creates a virtual environment and installs all Python dependencies.

---

## First-Time Setup

Run this **once** to create BigQuery tables, download Fama-French factors, and filter the universe:

```bash
python run_local.py --mode setup
```

What it does:
1. Creates all 12 BigQuery tables in your `output_results` dataset
2. Downloads FF 3-factor daily data from Ken French's website → writes to `ff_factors`
3. Skips ticker metadata (assumes `ticker_metadata` already exists in your BQ)
4. Runs universe quality filter → writes valid tickers to `filtered_universe`

Skip the (slow) historical residuals step:
```bash
python run_local.py --mode setup  # already skips historical by default
```

---

## Running the Pipeline

### Single Window (most common)
```bash
# Full pipeline: residuals → pairs → aggregation
python run_local.py \
  --window-start 2022-01-01 \
  --window-end 2022-12-31

# With 4 parallel workers for pair computation
python run_local.py \
  --window-start 2022-01-01 \
  --window-end 2022-12-31 \
  --workers 4
```

### Run Individual Steps
If a step fails partway through, you can re-run just that step:

```bash
# Step 1: Compute residuals only
python run_local.py \
  --window-start 2022-01-01 --window-end 2022-12-31 \
  --step residuals

# Step 2: Compute pairs only (residuals already in BQ)
python run_local.py \
  --window-start 2022-01-01 --window-end 2022-12-31 \
  --step pairs --workers 4

# Step 3: Aggregation only (pairs already in BQ)
python run_local.py \
  --window-start 2022-01-01 --window-end 2022-12-31 \
  --step aggregation

# Steps 2+3 together
python run_local.py \
  --window-start 2022-01-01 --window-end 2022-12-31 \
  --step pairs+agg --workers 4
```

### Latest Window (monthly update equivalent)
```bash
python run_local.py --mode latest --monthly
```

### Historical Backfill
Runs all windows defined by your `config.yaml` date range, one after another:

```bash
# Full backfill (slow — can take many hours for 2010–2025)
python run_local.py --mode backfill

# Backfill only residuals first (faster, then do pairs separately)
python run_local.py --mode backfill --step residuals

# Then backfill pairs with parallelism
python run_local.py --mode backfill --step pairs --workers 4

# Then backfill aggregation
python run_local.py --mode backfill --step aggregation

# Backfill a specific date range only
python run_local.py --mode backfill \
  --backfill-start 2020-01-01 --backfill-end 2023-12-31 \
  --workers 4
```

### Quarterly Batch (by year)
Equivalent to `quarterly_batches.sh`:

```bash
python run_local.py --mode quarterly --year 2022 --workers 4
```

---

## Performance Guide

### Pair computation is the bottleneck
With ~800 tickers, you have ~320K pairs × 5 lags × adaptive permutation.

| Workers | Partitions | Estimated time per window |
|---|---|---|
| 1 | 1 | ~8–12 hours |
| 4 | 16 | ~2–3 hours |
| 8 | 32 | ~1–1.5 hours |
| 16 | 64 | ~45 min |
| 90 | 360 | ~6–8 min (paw VM) |

Set `--workers` to your CPU core count minus a few, and `--partitions` to 4× workers for better load balancing:
```bash
# Check your core count
python3 -c "import os; print(os.cpu_count())"

# Use 7 workers on an 8-core machine
python run_local.py --window-start 2022-01-01 --window-end 2022-12-31 --workers 7 --partitions 28

# On the paw VM (96 CPUs)
python run_local.py --window-start 2022-01-01 --window-end 2022-12-31 --workers 90 --partitions 360 --log-dir /mnt/data1/logs
```

### Running on `paw` (UCD IDAV VM — 96 CPUs, 376 GB RAM)

See [VM_MIGRATION.md](VM_MIGRATION.md) for full details. Quick start:

```bash
# Clone to the large NVMe partition — /home is nearly full
cd /mnt/data1
git clone <repo-url> trading && cd trading
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full backfill with 90 workers
python run_local.py \
  --mode backfill \
  --workers 90 \
  --partitions 360 \
  --skip-synthetic \
  --log-dir /mnt/data1/logs
```

### Reducing the universe size (for testing)
Edit `config/config.yaml` to raise thresholds and reduce pair count:

```yaml
universe:
  min_market_cap: 1_000_000_000       # $1B (was $300M) → ~400 tickers
  min_median_dollar_volume: 20_000_000 # $20M (was $5M) → further reduction
```

With 400 tickers: ~80K pairs — roughly 4× faster.

### Reducing permutations (for testing/debugging)
```yaml
permutation:
  tier1_n: 50        # was 100
  tier1_cutoff: 0.30 # stop earlier
  tier2_n: 200       # was 500
  tier3_n: 500       # was 1000
```

---

## Configuration

All parameters live in `config/config.yaml`. Key settings to review before running locally:

```yaml
gcp:
  project_id: "capstone-487001"    # ← Your GCP project
  bq_dataset: "output_results"     # ← BigQuery dataset for results
  gcs_bucket: "capstone-bucket-487001"

universe:
  start_date: "2010-01-01"         # ← Reduce for faster local testing
  end_date: "2025-12-31"

compute:
  partition_count: 50              # ← Ignored locally; use --workers instead
```

For quick local testing, set a narrow date range:
```yaml
universe:
  start_date: "2020-01-01"
  end_date: "2022-12-31"
```

---

## Logs

Each run writes a timestamped log file:
```
pipeline_20250115_143022.log
```

Log level is INFO by default. The log goes to both stdout and the file.

---

## Common Errors

### `No valid tickers found`
The `filtered_universe` table is empty. Run setup first:
```bash
python run_local.py --mode setup
```

### `No residuals found for window`
The residuals step didn't run for this window yet. Run:
```bash
python run_local.py --window-start ... --window-end ... --step residuals
```

### `google.auth.exceptions.DefaultCredentialsError`
Not authenticated. Run:
```bash
gcloud auth application-default login
```

### `403 Access Denied` on BigQuery
Your account/ADC needs `roles/bigquery.dataEditor` and `roles/bigquery.jobUser` on the project.

### Pair job runs out of memory
Reduce the number of workers or the universe size. Each worker loads the full residuals pivot table into RAM. With 800 tickers × 252 days, this is ~1.6M floats ≈ ~13MB per worker, which is fine. If you have many more tickers, consider increasing available RAM or reducing `--workers`.

---

## Running Tests

```bash
# From the project root
pytest tests/test_core.py -v
```

Tests are purely computational (no BigQuery calls) and run offline.

---

## Project Structure

```
lead_lag_local/
├── run_local.py          ← MAIN ENTRY POINT — run this
├── install.sh            ← One-time environment setup
├── requirements.txt
├── config/
│   └── config.yaml       ← All parameters
├── src/                  ← All pipeline modules (unchanged from cloud version)
│   ├── bq_io.py
│   ├── bq_schema.py
│   ├── bootstrap.py
│   ├── config_loader.py  ← Updated: auto-detects config path
│   ├── dcor_engine.py
│   ├── fdr.py
│   ├── monte_carlo.py
│   ├── network.py
│   ├── oos_model.py
│   ├── permutation.py
│   ├── residuals.py
│   ├── stability.py
│   ├── synthetic.py
│   ├── universe.py
│   └── windows.py
├── jobs/
│   ├── pair_job.py       ← Same as cloud; called by run_local.py
│   └── aggregation_job.py ← Same as cloud; called by run_local.py
├── scripts/
│   └── setup.py          ← One-time BQ setup
└── tests/
    └── test_core.py
```

---

## Difference from Cloud Version

The only files changed from the original cloud version are:

1. **`src/config_loader.py`** — Path auto-detection updated to find `config/config.yaml` relative to the `src/` directory (works both in Cloud Run at `/app/` and locally).

2. **`run_local.py`** (new) — Replaces Cloud Workflows + Cloud Scheduler. Calls the same `pair_job.py` and `aggregation_job.py` functions directly.

3. **`install.sh`** (new) — Sets up the local Python environment.

4. **`README_LOCAL.md`** (this file).

All `src/`, `jobs/`, and `scripts/` files are identical to the cloud version. If you update any of those in the cloud repo, copy them here too.
