# Lead-Lag Signal Pipeline

A production-grade quantitative signal discovery system that identifies statistically robust directional lead-lag relationships between equity pairs using distance correlation (dCor), adaptive permutation testing, and economic validation via OOS trading strategy performance.

---

## Architecture Overview

```
Cloud Scheduler (1st of month)
    ↓
Cloud Workflows
    ├── Residuals Job (1 job)
    ├── Pair Jobs × 50 (parallel Cloud Run)
    └── Aggregation Job (1 job)
        ├── FDR correction
        ├── OOS strategy evaluation
        ├── Stability metrics
        ├── Model weights (quarterly refit)
        ├── final_network update
        ├── Network centrality
        ├── Monte Carlo cones
        └── Synthetic health check
            ↓
        BigQuery
            ↓
        FastAPI → React Frontend
```

---

## Repository Structure

```
/config
    config.yaml                 ← All parameters (edit before deploying)
/src
    config_loader.py            ← Config access utility
    bq_io.py                    ← All BigQuery read/write operations
    bq_schema.py                ← Table schemas + creation
    universe.py                 ← Quality filtering
    windows.py                  ← Rolling window generation
    residuals.py                ← FF factor download + OLS residualization
    dcor_engine.py              ← Distance correlation + sharpness
    permutation.py              ← Adaptive 3-tier block permutation
    fdr.py                      ← Benjamini-Hochberg FDR correction
    stability.py                ← Cross-window stability metrics
    oos_model.py                ← OOS trading strategy evaluation
    bootstrap.py                ← OOS regression + bootstrap CI + signal strength
    network.py                  ← Directed graph + centrality
    monte_carlo.py              ← Monte Carlo PnL cones
    synthetic.py                ← Fixed-seed health check
/jobs
    pair_job.py                 ← Cloud Run parallel worker
    aggregation_job.py          ← Cloud Run aggregation orchestrator
/workflows
    monthly_pipeline.yaml       ← Cloud Workflows definition
/api
    main.py                     ← FastAPI backend (all endpoints)
/scripts
    setup.py                    ← One-time initialization
    deploy.sh                   ← GCP infrastructure deployment
    ingest_ticker_metadata.py   ← Yahoo Finance sector/metadata ingestion
/tests
    test_core.py                ← Unit tests for all core modules
```

---

## Setup Instructions

### 1. Configure
Edit `config/config.yaml`:
- Set `gcp.project_id` and `gcp.gcs_bucket`
- Adjust `universe` thresholds if needed
- Set `universe.start_date` / `end_date`

### 2. Deploy GCP Infrastructure
```bash
bash scripts/deploy.sh
```

### 3. Run Initial Setup
```bash
python -m scripts.setup
```
This:
- Creates all BigQuery tables
- Downloads Fama-French 3-factor data
- Ingests ticker metadata from Yahoo Finance
- Runs universe quality filter
- Computes historical residuals

### 4. Run Historical Backfill
Trigger pair computation for all historical windows. This is the expensive one-time operation:
```bash
# Trigger Cloud Run jobs for each historical window
# (automate via script or trigger manually per window)
gcloud run jobs execute pair-job --region=us-central1 \
  --update-env-vars=WINDOW_START=2010-01-01,WINDOW_END=2011-01-01,PARTITION_ID=0,NUM_PARTITIONS=50,RUN_ID=backfill
```

### 5. Run Initial Aggregation
```bash
gcloud run jobs execute aggregation-job --region=us-central1 \
  --update-env-vars=WINDOW_START=2010-01-01,WINDOW_END=2011-01-01,RUN_ID=backfill,IS_MONTHLY_UPDATE=false
```

### 6. Deploy API
```bash
gcloud run deploy lead-lag-api \
  --image=gcr.io/YOUR_PROJECT/quant-pipeline:latest \
  --region=us-central1 \
  --command=python \
  --args="-m,uvicorn,api.main:app,--host,0.0.0.0,--port,8080"
```

---

## Algorithm

### Signal Discovery
1. **Residualization**: OLS regression of log returns on Fama-French 3-factor returns per rolling 252-day window (63-day step)
2. **dCor**: Distance correlation at lags 1–5 for all ~320K pairs
3. **Adaptive Permutation (3-tier)**:
   - Tier 1: 100 block permutations → stop if p > 0.20
   - Tier 2: extend to 500 → stop if p > 0.10
   - Tier 3: extend to 1000 (hard ceiling)
4. **Best Lag Selection**: Keep lag with highest significant dCor per pair
5. **FDR Control**: Benjamini-Hochberg at 5% level

### Stability Features (X_ij)
| Feature | Description |
|---|---|
| `mean_dcor` | Mean dCor across significant windows |
| `variance_dcor` | Variance of dCor across windows |
| `frequency` | Fraction of windows where pair was significant |
| `half_life` | Exponential decay fit to dCor over time (days) |
| `sharpness` | Entropy-based concentration of dCor at a single lag |

### Economic Validation
- **OOS Strategy**: Long/short B when rolling z-score of A exceeds ±1 (60-day lookback, 1-day hold, 10bps round-trip cost)
- **OOS Metric (Y_ij)**: Global net Sharpe across all concatenated OOS windows (~1000+ days)
- **OOS Regression**: Y_ij ~ X_ij features → learns β weights
- **Bootstrap CI**: 1000 resamplings for confidence intervals on β

### Outputs
- **`predicted_sharpe`**: β · X_ij (raw economic prediction)
- **`signal_strength`**: Winsorized 0–100 normalization of predicted Sharpe

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /pairs/top` | Top N pairs by signal strength |
| `GET /pairs/search?ticker=AAPL` | Pairs involving a ticker |
| `GET /pairs/{ti}/{tj}` | Pair detail |
| `GET /pairs/by-sector?sector=Technology` | Sector filter |
| `GET /network` | Graph JSON (nodes + edges) |
| `GET /network/centrality` | Top central nodes |
| `GET /features/importance` | β weights with CI bands |
| `GET /charts/cumulative-return` | Monte Carlo cone data |
| `GET /charts/decile-performance` | Decile vs OOS Sharpe |
| `GET /charts/lag-distribution` | Lag histogram |
| `GET /charts/centrality-rolling` | Centrality persistence |
| `GET /meta/config` | Public config |
| `GET /meta/last-update` | Last pipeline run info |

### Query Parameters (pairs/top)
```
?n=100&min_score=60&sector=Technology&lag=2&intra_sector_only=false&inter_sector_only=true
```

---

## Monthly Pipeline (Automated)

Runs on **1st of every month** via Cloud Scheduler → Cloud Workflows:

1. ✅ Append new price data to `daily_prices`
2. ✅ Compute residuals for newest window only
3. ✅ Run 50 parallel pair jobs (~320K pairs)
4. ✅ Apply FDR to new window
5. ✅ Evaluate OOS strategy on new OOS window
6. ✅ Append OOS returns, update global Sharpe
7. ✅ Recompute stability metrics
8. ✅ Apply frozen β (no refit)
9. ✅ Update `final_network`
10. ✅ Recompute network centrality
11. ✅ Run Monte Carlo cones
12. ✅ Synthetic health check
13. ✅ Log run

**Quarterly (Jan, Apr, Jul, Oct)**: β weights are refit from scratch.

---

## Configuration Reference

Key parameters in `config/config.yaml`:

```yaml
universe:
  min_market_cap: 300_000_000       # Raise to reduce universe size
  min_median_dollar_volume: 5_000_000
  min_trading_day_coverage: 0.95

permutation:
  tier1_n: 100
  tier1_cutoff: 0.20
  tier2_n: 500
  tier2_cutoff: 0.10
  tier3_n: 1000

compute:
  partition_count: 50               # Increase for faster runs

fdr:
  alpha: 0.05                       # Lower = fewer but cleaner signals
```

---

## Synthetic Health Check

Runs monthly with fixed seed (42). Tests pipeline integrity:

- 50 planted pairs with threshold-regime nonlinear lag-2 relationships
- 450 null pairs (some with GARCH-like volatility clustering)

**Alerts if**:
- True Positive Rate < 70%
- False Positive Rate > 10%

Results logged to `synthetic_health_log` in BigQuery.

---

## Running Tests

```bash
pytest tests/test_core.py -v
```

Tests cover: dCor detection, sharpness, block permutation, FDR, half-life estimation, OOS strategy, bootstrap normalization, and synthetic data generation.

---

## Cost Control

- **Only newest window computed monthly** — no historical recomputation
- **Adaptive permutation** — ~70-80% pairs stop at tier 1 (100 perms)
- **Budget guard** — alerts if >10% of pairs hit tier 3
- **Residuals precomputed** — never recomputed inside pair loops
- **Lags 1-5 only** — best lag kept, single edge per pair for FDR

Estimated monthly cost: < $50 for 320K pairs × 5 lags with adaptive permutation.
