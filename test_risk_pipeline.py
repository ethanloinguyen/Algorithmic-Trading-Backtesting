"""
Clustering quality + speed comparison: hierarchical (optimized) vs hierarchical_original.

Runs both implementations against the same portfolio using a shared BigQuery client,
then prints a side-by-side report covering:
  - Wall-clock time for the clustering step
  - Selected k and silhouette score
  - Sectors covered
  - Per-sector stock picks and avg_dcor values

Run from the repo root:
    python test_risk_pipeline.py
"""

import sys, pathlib, time, io, contextlib, re, importlib.util
sys.path.insert(0, str(pathlib.Path("model")))
sys.path.insert(0, str(pathlib.Path("backend")))

from google.cloud import bigquery

# ── Load both modules by file path to avoid sys.path conflicts ────────────────

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_MODEL_DIR = pathlib.Path("model")
hier_opt  = _load("hierarchical",          _MODEL_DIR / "hierarchical.py")
hier_orig = _load("hierarchical_original", _MODEL_DIR / "hierarchical_original.py")

# ── Config ────────────────────────────────────────────────────────────────────

PORTFOLIO = ["AAPL", "MSFT", "GOOGL"]

DIV  = "=" * 72
DIV2 = "-" * 72

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_and_capture(module, portfolio, client, label):
    """Run run_clustering, capture stdout, time the call, parse key metrics."""
    buf = io.StringIO()
    t0  = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        recs = module.run_clustering(user_portfolio=portfolio, bq_client=client)
    elapsed = time.perf_counter() - t0
    output  = buf.getvalue()

    # Parse "✅ Final k = N"
    k_match = re.search(r"Final k\s*=\s*(\d+)", output)
    k_final = int(k_match.group(1)) if k_match else None

    # Parse silhouette from "sil=X.XXXX" on the "Selected k=" line
    sil_match = re.search(r"Selected k=\d+.*?sil=([-+]?\d+\.\d+)", output)
    sil_final = float(sil_match.group(1)) if sil_match else float("nan")

    print(f"\n{'━' * 72}")
    print(f"  {label}")
    print("━" * 72)
    print(output)

    return recs, k_final, sil_final, elapsed


def _compare(label_a, recs_a, k_a, sil_a, t_a,
             label_b, recs_b, k_b, sil_b, t_b):

    sectors_a = set(recs_a["sector"].tolist())
    sectors_b = set(recs_b["sector"].tolist())
    dcor_a    = dict(zip(recs_a["sector"], recs_a["avg_dcor_to_portfolio"]))
    dcor_b    = dict(zip(recs_b["sector"], recs_b["avg_dcor_to_portfolio"]))

    print(f"\n{DIV}")
    print("  SPEED")
    print(DIV2)
    print(f"  {label_a:<32}  {t_a:>7.1f}s")
    print(f"  {label_b:<32}  {t_b:>7.1f}s")
    speedup = t_b / t_a if t_a > 0 else float("inf")
    print(f"  Speedup ({label_a} vs {label_b}):  {speedup:.2f}x faster")

    print(f"\n{DIV}")
    print("  CLUSTERING QUALITY")
    print(DIV2)
    print(f"  {'Metric':<30}  {label_a:>16}  {label_b:>16}")
    print(DIV2)
    print(f"  {'Selected k':<30}  {str(k_a):>16}  {str(k_b):>16}")
    print(f"  {'Silhouette @ selected k':<30}  {sil_a:>+16.4f}  {sil_b:>+16.4f}")
    print(f"  {'Sectors covered':<30}  {len(sectors_a):>16}  {len(sectors_b):>16}")

    print(f"\n{DIV}")
    print("  PER-SECTOR PICKS")
    print(DIV2)
    print(f"  {'':1}  {'Sector':<28}  {label_a:>20}  {label_b:>20}")
    print(DIV2)
    for sec in sorted(sectors_a | sectors_b):
        row_a = recs_a[recs_a["sector"] == sec]
        row_b = recs_b[recs_b["sector"] == sec]
        a_str = f"{row_a.iloc[0]['stock']}  {row_a.iloc[0]['avg_dcor_to_portfolio']:.4f}" if not row_a.empty else "—"
        b_str = f"{row_b.iloc[0]['stock']}  {row_b.iloc[0]['avg_dcor_to_portfolio']:.4f}" if not row_b.empty else "—"
        match = "✓" if (not row_a.empty and not row_b.empty
                        and row_a.iloc[0]["stock"] == row_b.iloc[0]["stock"]) else " "
        print(f"  {match}  {sec:<28}  {a_str:>20}  {b_str:>20}")

    only_a = sectors_a - sectors_b
    only_b = sectors_b - sectors_a
    if only_a:
        print(f"\n  Sectors only in {label_a}: {sorted(only_a)}")
    if only_b:
        print(f"  Sectors only in {label_b}: {sorted(only_b)}")

    shared = sectors_a & sectors_b
    if shared:
        deltas    = {s: abs(dcor_a[s] - dcor_b[s]) for s in shared}
        avg_delta = sum(deltas.values()) / len(deltas)
        max_delta = max(deltas.values())
        print(f"\n  Avg |dcor delta| across shared sectors:  {avg_delta:.4f}")
        print(f"  Max |dcor delta| across shared sectors:  {max_delta:.4f}")

    sil_delta = abs(sil_a - sil_b)
    verdict = (
        "negligible — quality equivalent" if sil_delta < 0.02 else
        "minor — acceptable"              if sil_delta < 0.05 else
        "notable — consider tuning optimized params"
    )
    print(f"\n  Silhouette delta: {sil_delta:.4f}  ({verdict})")
    print(DIV)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(DIV)
    print("  Clustering comparison: optimized vs original")
    print(f"  Portfolio: {PORTFOLIO}")
    print(DIV)

    # Shared client — avoids double credential load skewing timing
    client = bigquery.Client(project="capstone-487001")

    print("\n>>> Running OPTIMIZED implementation ...")
    recs_opt, k_opt, sil_opt, t_opt = _run_and_capture(
        hier_opt, PORTFOLIO, client, "OPTIMIZED (hierarchical.py)"
    )

    print("\n>>> Running ORIGINAL implementation ...")
    recs_orig, k_orig, sil_orig, t_orig = _run_and_capture(
        hier_orig, PORTFOLIO, client, "ORIGINAL (hierarchical_original.py)"
    )

    _compare(
        "optimized", recs_opt,  k_opt,  sil_opt,  t_opt,
        "original",  recs_orig, k_orig, sil_orig, t_orig,
    )


if __name__ == "__main__":
    main()
