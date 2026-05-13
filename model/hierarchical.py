"""
Stock Portfolio Diversification — Multi-Feature K-Medoids (v9)
==============================================================
Reads candidate stocks from a BigQuery table, builds a feature matrix,
runs a K-Medoids sweep, selects k via elbow + sector-diversity, and
outputs one recommended stock per sector.

Requirements
------------
    pip install google-cloud-bigquery pandas numpy scikit-learn matplotlib seaborn db-dtypes

Authentication
--------------
Set the environment variable before running:
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
Or run:  gcloud auth application-default login

Importable API
--------------
    from hierarchical import run_clustering
    recs = run_clustering(['AAPL', 'MSFT', 'GOOGL'])
    # returns a DataFrame: [sector, stock, cluster, is_medoid,
    #                        avg_dcor_to_portfolio, mean_intra_dist,
    #                        n_sector_candidates, cluster_size]
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from google.cloud import bigquery
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# =============================================================================
# 1 · K-Medoids (pure NumPy)
# =============================================================================

class KMedoidsPAM:
    def __init__(self, n_clusters=8, max_iter=300, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def _kpp_init(self, D, rng):
        n = D.shape[0]
        medoids = [rng.randint(0, n)]
        for _ in range(1, self.n_clusters):
            d2 = D[:, medoids].min(axis=1) ** 2
            s = d2.sum()
            medoids.append(rng.choice(n, p=d2 / s if s > 0 else np.ones(n) / n))
        return np.array(medoids)

    @staticmethod
    def _assign(D, m):
        return np.argmin(D[:, m], axis=1)

    @staticmethod
    def _inertia(D, labels, m):
        return float(sum(D[i, m[labels[i]]] for i in range(len(labels))))

    def _fit_once(self, D, rng):
        m = self._kpp_init(D, rng)
        for _ in range(self.max_iter):
            labels = self._assign(D, m)
            new_m = m.copy()
            for cid in range(self.n_clusters):
                mask = np.where(labels == cid)[0]
                if len(mask):
                    new_m[cid] = mask[D[np.ix_(mask, mask)].sum(axis=1).argmin()]
            if np.array_equal(np.sort(new_m), np.sort(m)):
                break
            m = new_m
        labels = self._assign(D, m)
        return labels, m, self._inertia(D, labels, m)

    def fit(self, D):
        rng = np.random.RandomState(self.random_state)
        best = np.inf
        for _ in range(self.n_init):
            inner = np.random.RandomState(rng.randint(0, 2 ** 31))
            labels, m, inertia = self._fit_once(D, inner)
            if inertia < best:
                best = inertia
                self.labels_ = labels
                self.medoid_indices_ = m
                self.inertia_ = inertia
        return self


# =============================================================================
# 2 · Default Configuration
# =============================================================================

GCP_PROJECT    = 'capstone-487001'
BQ_TABLE       = f'{GCP_PROJECT}.output_results.final_network'
COL_STOCK_A    = 'ticker_i'
COL_STOCK_B    = 'ticker_j'
COL_DCOR       = 'mean_dcor'
USER_PORTFOLIO = ['AAPL', 'MSFT', 'GOOGL']

NUMERIC_FEATURES = [
    'mean_dcor', 'variance_dcor', 'best_lag', 'frequency', 'half_life',
    'sharpness', 'predicted_sharpe', 'signal_strength',
    'oos_sharpe_net', 'oos_dcor', 'centrality_i',
]
USE_SECTOR                 = True
SECTOR_WEIGHT              = 1.0
CORRELATION_DROP_THRESHOLD = 0.85
PCA_COMPONENTS             = None

MIN_CLUSTER_SIZE     = 8
K_MIN                = 3
N_INIT               = 25
MAX_CLUSTER_FRACTION = 0.25
ELBOW_SENSITIVITY    = 0.45
MIN_SECTORS_TARGET   = 6
N_CLUSTERS           = None  # None = objective selection


# =============================================================================
# 3 · BigQuery helpers
# =============================================================================

def build_candidate_pool(portfolio, bq_table, dcor_threshold, client):
    tickers_sql = ', '.join(f"'{t}'" for t in portfolio)
    query = f"""
    WITH normalised AS (
        SELECT {COL_STOCK_A} AS portfolio_stock, {COL_STOCK_B} AS candidate_stock, {COL_DCOR} AS dcor
        FROM `{bq_table}`
        WHERE {COL_STOCK_A} IN ({tickers_sql}) AND {COL_STOCK_B} NOT IN ({tickers_sql})
          AND {COL_DCOR} <= {dcor_threshold}
        UNION ALL
        SELECT {COL_STOCK_B}, {COL_STOCK_A}, {COL_DCOR}
        FROM `{bq_table}`
        WHERE {COL_STOCK_B} IN ({tickers_sql}) AND {COL_STOCK_A} NOT IN ({tickers_sql})
          AND {COL_DCOR} <= {dcor_threshold}
    ),
    aggregated AS (
        SELECT candidate_stock, COUNT(DISTINCT portfolio_stock) AS n_links,
               AVG(dcor) AS avg_dcor_to_portfolio, MIN(dcor) AS min_dcor_to_portfolio
        FROM normalised GROUP BY candidate_stock
    )
    SELECT * FROM aggregated WHERE n_links = {len(portfolio)}
    ORDER BY avg_dcor_to_portfolio ASC
    """
    df = client.query(query).to_dataframe()
    print(f'Candidate pool: {len(df):,} stocks')
    return df


def fetch_feature_rows_bidirectional(stocks, bq_table, client,
                                      numeric_features=None, use_sector=True):
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    tickers_sql = ', '.join(f"'{t}'" for t in stocks)
    feat_i = ', '.join(numeric_features)
    sector_i = ', sector_i' if use_sector else ''
    feat_j_parts = ['centrality_j AS centrality_i' if f == 'centrality_i' else f
                    for f in numeric_features]
    feat_j = ', '.join(feat_j_parts)
    sector_j = ', sector_j AS sector_i' if use_sector else ''
    query = f"""
    SELECT {COL_STOCK_A} AS stock, {feat_i}{sector_i} FROM `{bq_table}` WHERE {COL_STOCK_A} IN ({tickers_sql})
    UNION ALL
    SELECT {COL_STOCK_B} AS stock, {feat_j}{sector_j} FROM `{bq_table}` WHERE {COL_STOCK_B} IN ({tickers_sql})
    """
    print('Fetching bidirectional feature rows ...')
    df = client.query(query).to_dataframe()
    print(f'  -> {len(df):,} rows for {df["stock"].nunique()} stocks  '
          f'(avg {len(df)/df["stock"].nunique():.0f} rows/stock)')
    return df


# =============================================================================
# 4 · Feature matrix
# =============================================================================

def build_feature_matrix(feature_rows, candidate_stocks,
                          numeric_features=None, use_sector=True, sector_weight=1.0):
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    numeric_agg = feature_rows.groupby('stock')[numeric_features].median().reindex(candidate_stocks)
    numeric_df = pd.DataFrame(
        SimpleImputer(strategy='median').fit_transform(numeric_agg),
        index=candidate_stocks, columns=numeric_features
    )
    if use_sector:
        sector_mode = (
            feature_rows.groupby('stock')['sector_i']
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown')
            .reindex(candidate_stocks, fill_value='Unknown')
        )
        sector_dummies = pd.get_dummies(sector_mode, prefix='sector') * sector_weight
        print(f'Sectors: {sorted(sector_mode.unique())}')
        out = pd.concat([numeric_df, sector_dummies], axis=1)
    else:
        out = numeric_df
    print(f'Feature matrix: {out.shape[0]} x {out.shape[1]}')
    return out


# =============================================================================
# 5 · Feature selection & scaling helpers
# =============================================================================

def drop_correlated_features(df, threshold=0.85):
    numeric_cols = [c for c in df.columns if not c.startswith('sector_')]
    sector_cols  = [c for c in df.columns if c.startswith('sector_')]
    corr  = df[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept    = [c for c in numeric_cols if c not in to_drop]
    if to_drop:
        print(f'Dropped ({len(to_drop)}): {to_drop}')
    else:
        print(f'No features dropped (all correlations <= {threshold})')
    print(f'Kept numeric ({len(kept)}): {kept}')
    return df[kept + sector_cols], kept


# =============================================================================
# 6 · K-Medoids sweep
# =============================================================================

def balance_cv(labels):
    c = np.array(list(Counter(labels).values()), dtype=float)
    return c.std() / c.mean()


def run_full_sweep(D, k_min, k_max, n_init, min_size, max_frac, rs=42):
    results  = {}
    n_stocks = D.shape[0]
    print(f'Sweep k={k_min}...{k_max}  min_size={min_size}  n_init={n_init}')
    print(f'{"k":>4}  {"valid":>6}  {"inertia":>10}  {"silhouette":>11}  {"bal_cv":>8}  {"max_cl":>7}  sizes')
    print('-' * 90)
    for k in range(k_min, k_max + 1):
        km     = KMedoidsPAM(n_clusters=k, max_iter=300, n_init=n_init, random_state=rs)
        km.fit(D)
        labels   = km.labels_
        sizes    = sorted(Counter(labels).values())
        valid    = min(sizes) >= min_size
        sil      = silhouette_score(D, labels, metric='precomputed') if valid else float('nan')
        bal      = balance_cv(labels) if valid else float('nan')
        max_cl   = max(sizes)
        oversized = max_cl > n_stocks * max_frac
        results[k] = {
            'labels'    : labels,
            'medoids'   : km.medoid_indices_,
            'inertia'   : km.inertia_,
            'sil'       : sil,
            'bal_cv'    : bal,
            'valid'     : valid,
            'sizes'     : sizes,
            'oversized' : oversized,
        }
        flag  = '  OK' if valid else '   X'
        ss    = f'{sil:>+11.4f}' if valid else '          -'
        bs    = f'{bal:>8.3f}'   if valid else '       -'
        warn  = ' ⚠' if oversized else '  '
        print(f'{k:>4}  {flag}  {km.inertia_:>10.4f}  {ss}  {bs}  {max_cl:>5}{warn}  {sizes}')
    return results


# =============================================================================
# 7 · k selection helpers
# =============================================================================

def detect_elbow(sweep, sensitivity):
    ks       = [k for k in sorted(sweep.keys()) if sweep[k]['valid']]
    inertias = [sweep[k]['inertia'] for k in ks]
    if len(ks) < 3:
        return ks[0], ks, inertias, [], []
    drops  = [max(0.0, inertias[i] - inertias[i+1]) for i in range(len(inertias)-1)]
    accels = [drops[i] - drops[i+1] for i in range(len(drops)-1)]
    e_accel = ks[1 + int(np.argmax(accels))]
    e_sens  = next(
        (ks[i] for i in range(1, len(drops))
         if drops[i-1] > 0 and drops[i] / drops[i-1] < sensitivity), None
    )
    elbow_k = max(e_accel, e_sens) if e_sens else e_accel
    print(f'Stage A — Elbow detection:')
    print(f'  Valid k range            : {min(ks)}–{max(ks)}')
    print(f'  Max-acceleration method  : k={e_accel}')
    print(f'  Sensitivity ({sensitivity}) method: k={e_sens}')
    print(f'  Floor (conservative)     : k={elbow_k}')
    return elbow_k, ks, inertias, drops, accels


# =============================================================================
# 8 · Main pipeline (importable entry point)
# =============================================================================

def run_clustering(
    user_portfolio: list,
    gcp_project: str = GCP_PROJECT,
    bq_table: str = BQ_TABLE,
    dcor_threshold: float = 0.25,
    numeric_features: list = None,
    use_sector: bool = USE_SECTOR,
    sector_weight: float = SECTOR_WEIGHT,
    correlation_drop_threshold: float = CORRELATION_DROP_THRESHOLD,
    pca_components: int = None,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    k_min: int = K_MIN,
    n_init: int = N_INIT,
    max_cluster_fraction: float = MAX_CLUSTER_FRACTION,
    elbow_sensitivity: float = ELBOW_SENSITIVITY,
    min_sectors_target: int = MIN_SECTORS_TARGET,
    n_clusters: int = None,
    bq_client=None,
    save_csv: bool = False,
    save_plot: bool = False,
) -> pd.DataFrame:
    """
    Run the full hierarchical K-Medoids clustering pipeline.

    Parameters
    ----------
    user_portfolio : list of ticker symbols already in the user's portfolio
    bq_client      : optional pre-existing BigQuery client (reused when called
                     from the backend to avoid repeated credential loading)
    save_csv       : write recommended_stocks.csv to the working directory
    save_plot      : render and save the t-SNE sector star graph

    Returns
    -------
    DataFrame with columns:
        sector, stock, cluster, is_medoid, avg_dcor_to_portfolio,
        mean_intra_dist, n_sector_candidates, cluster_size
    Raises ValueError if the quality check fails.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    client = bq_client or bigquery.Client(project=gcp_project)

    # ── Candidate pool ────────────────────────────────────────────────────────
    candidate_df     = build_candidate_pool(user_portfolio, bq_table, dcor_threshold, client)
    candidate_stocks = candidate_df['candidate_stock'].tolist()

    k_max_eff = max(k_min + 1, len(candidate_stocks) // min_cluster_size)
    print(f'Auto K_MAX = {k_max_eff}  (pool={len(candidate_stocks)} / min_size={min_cluster_size})')

    # ── Feature matrix ────────────────────────────────────────────────────────
    feature_rows   = fetch_feature_rows_bidirectional(candidate_stocks, bq_table, client,
                                                       numeric_features, use_sector)
    feature_matrix = build_feature_matrix(feature_rows, candidate_stocks,
                                           numeric_features, use_sector, sector_weight)
    sector_map = (
        feature_rows.groupby('stock')['sector_i']
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown')
    )

    # ── Feature selection & scaling ───────────────────────────────────────────
    feature_matrix_sel, kept_numeric = drop_correlated_features(feature_matrix,
                                                                  correlation_drop_threshold)
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(feature_matrix_sel.values)

    if pca_components is not None:
        pca_t    = PCA(n_components=min(pca_components, X_scaled.shape[1] - 1),
                       whiten=True, random_state=42)
        X_scaled = pca_t.fit_transform(X_scaled)
        print(f'PCA: {X_scaled.shape[1]} components, {pca_t.explained_variance_ratio_.sum():.1%} variance')

    D = pairwise_distances(X_scaled, metric='euclidean').astype(np.float64)
    D /= D.max()

    # ── Sweep ─────────────────────────────────────────────────────────────────
    sweep = run_full_sweep(D, k_min, k_max_eff, n_init, min_cluster_size, max_cluster_fraction)

    # ── Inner helpers (close over local state) ────────────────────────────────

    def _build_cluster_df_for_k(k):
        labels  = sweep[k]['labels']
        medoids = sweep[k]['medoids']
        cdf = (
            pd.DataFrame({'stock': candidate_stocks, 'cluster': labels + 1})
            .merge(candidate_df[['candidate_stock', 'avg_dcor_to_portfolio', 'min_dcor_to_portfolio']],
                   left_on='stock', right_on='candidate_stock', how='left')
        )
        stock_feats = feature_matrix[kept_numeric].copy()
        cdf = cdf.merge(stock_feats, left_on='stock', right_index=True, how='left')
        if use_sector:
            cdf['sector'] = cdf['stock'].map(sector_map)
        cdf['is_medoid'] = cdf['stock'].isin([candidate_stocks[i] for i in medoids])
        return cdf

    def _pick_best_per_sector(cluster_df):
        dist_df = pd.DataFrame(D, index=candidate_stocks, columns=candidate_stocks)
        intra_series = {}
        for cid, group in cluster_df.groupby('cluster'):
            members    = group['stock'].tolist()
            sub_dist   = dist_df.loc[members, members]
            mean_intra = sub_dist.mean(axis=1)
            for stock, val in mean_intra.items():
                intra_series[stock] = val

        cdf = cluster_df.copy()
        cdf['mean_intra_dist'] = cdf['stock'].map(intra_series)

        records = []
        sectors = sorted([s for s in cdf['sector'].dropna().unique() if s != 'Unknown'])
        for sector in sectors:
            sector_stocks = cdf[cdf['sector'] == sector].copy()
            if sector_stocks.empty:
                continue
            sector_stocks['avg_dcor_to_portfolio'] = sector_stocks['avg_dcor_to_portfolio'].fillna(dcor_threshold)
            sector_stocks['mean_intra_dist']        = sector_stocks['mean_intra_dist'].fillna(
                cdf['mean_intra_dist'].median()
            )
            best = sector_stocks.sort_values(
                ['avg_dcor_to_portfolio', 'mean_intra_dist'], ascending=[True, True]
            ).iloc[0]
            records.append({
                'sector'               : sector,
                'stock'                : best['stock'],
                'cluster'              : int(best['cluster']),
                'is_medoid'            : bool(best['is_medoid']),
                'avg_dcor_to_portfolio': float(best['avg_dcor_to_portfolio']),
                'mean_intra_dist'      : float(best['mean_intra_dist']),
                'n_sector_candidates'  : len(sector_stocks),
                'cluster_size'         : int((cdf['cluster'] == best['cluster']).sum()),
            })
        return pd.DataFrame(records).sort_values('avg_dcor_to_portfolio').reset_index(drop=True)

    def _select_k_by_sector_diversity(elbow_k):
        valid_ks = [k for k in sorted(sweep) if sweep[k]['valid'] and k >= elbow_k]
        if not valid_ks:
            valid_ks = [k for k in sorted(sweep) if sweep[k]['valid']]
            print('  WARNING: No valid k >= elbow floor — using all valid k')
        print(f'\nStage B — Sector diversity across valid k >= {elbow_k}:')
        div = {}
        for k in valid_ks:
            cdf  = _build_cluster_df_for_k(k)
            recs = _pick_best_per_sector(cdf)
            secs = recs['sector'].tolist()
            div[k] = {
                'n_sectors' : len(secs),
                'sectors'   : secs,
                'n_picks'   : len(recs),
                'sil'       : sweep[k]['sil'],
                'oversized' : sweep[k]['oversized'],
            }
            marker = ' <-- meets target' if len(secs) >= min_sectors_target else ''
            print(f'{k:>4}  {len(secs):>8}  {len(recs):>6}  '
                  f'{sweep[k]["sil"]:>+11.4f}  '
                  f'{"  ⚠" if sweep[k]["oversized"] else "   ":>10}  {sorted(secs)}{marker}')
        best_k = max(div, key=lambda k: (div[k]['n_sectors'], div[k]['sil']))
        r = div[best_k]
        print(f'\n  -> Selected k={best_k}  '
              f'({r["n_sectors"]} sectors, {r["n_picks"]} picks, sil={r["sil"]:.4f})')
        return best_k, div

    # ── k selection ───────────────────────────────────────────────────────────
    if n_clusters is None:
        print('=' * 60)
        elbow_k, *_ = detect_elbow(sweep, elbow_sensitivity)
        print('=' * 60)
        n_clusters_final, _ = _select_k_by_sector_diversity(elbow_k)
    else:
        print(f'N_CLUSTERS manually pinned to {n_clusters} — skipping objective selection')
        n_clusters_final = n_clusters

    print(f'\n✅ Final k = {n_clusters_final}')

    # ── Final recommendations ─────────────────────────────────────────────────
    cluster_df      = _build_cluster_df_for_k(n_clusters_final)
    recommendations = _pick_best_per_sector(cluster_df)

    # ── Quality check ─────────────────────────────────────────────────────────
    failures = []
    sec_counts = recommendations['sector'].value_counts()
    for sec, cnt in sec_counts.items():
        if cnt != 1:
            failures.append(f'{sec} has {cnt} stocks (expected 1)')
    n_sectors = len(recommendations)
    if n_sectors < min_sectors_target:
        failures.append(f'Only {n_sectors} sectors < target {min_sectors_target}')
    for _, row in recommendations[recommendations['avg_dcor_to_portfolio'] > dcor_threshold].iterrows():
        failures.append(f'{row["stock"]} exceeds dcor ceiling {dcor_threshold}')
    if failures:
        raise ValueError(f'Clustering quality check failed: {failures}')

    print(f'STATUS: ✅ ALL CHECKS PASSED — {n_sectors} sectors | k={n_clusters_final} '
          f'| sil={sweep[n_clusters_final]["sil"]:.4f}')

    # ── Optional outputs ──────────────────────────────────────────────────────
    if save_csv:
        recommendations.to_csv('recommended_stocks.csv', index=False)
        print('Saved: recommended_stocks.csv')

    if save_plot:
        coords = TSNE(n_components=2, random_state=42,
                      perplexity=min(30, len(candidate_stocks) - 1),
                      init='pca', max_iter=1000).fit_transform(X_scaled)

        rec_tickers    = set(recommendations['stock'])
        sectors_sorted = sorted(recommendations['sector'].unique())
        sector_palette = {s: plt.cm.tab20(i / max(len(sectors_sorted) - 1, 1))
                          for i, s in enumerate(sectors_sorted)}

        fig, ax = plt.subplots(figsize=(13, 9))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')

        for idx, ticker in enumerate(candidate_stocks):
            if ticker not in rec_tickers:
                ax.scatter(coords[idx, 0], coords[idx, 1],
                           color='#cccccc', s=25, alpha=0.4, zorder=1)

        for _, row in recommendations.iterrows():
            idx   = candidate_stocks.index(row['stock'])
            color = sector_palette[row['sector']]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       color=color, s=400, marker='*',
                       edgecolors='black', linewidths=0.8, zorder=5)
            ax.annotate(
                f"{row['stock']}\n({row['sector']})",
                xy=(coords[idx, 0], coords[idx, 1]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=color)
            )

        ax.set_title(
            f'Sector Star Graph — k={n_clusters_final}  |  {n_sectors} recommended stocks  '
            f'(★ = sector pick,  · = candidate pool)',
            fontsize=13, fontweight='bold', pad=14
        )
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        handles = [
            plt.scatter([], [], marker='*', s=150, color=sector_palette[s],
                        edgecolors='black', linewidths=0.5, label=s)
            for s in sectors_sorted
        ]
        ax.legend(handles=handles, title='Sector', bbox_to_anchor=(1.01, 1),
                  loc='upper left', fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.savefig('sector_star_graph.png', dpi=150, bbox_inches='tight')
        plt.show()
        print('Saved: sector_star_graph.png')

    return recommendations


# =============================================================================
# 9 · Standalone entry point (unchanged behaviour when run as a script)
# =============================================================================

if __name__ == '__main__':
    recs = run_clustering(
        USER_PORTFOLIO,
        save_csv=True,
        save_plot=True,
    )

    sil_val = None  # available inside run_clustering — print summary from DataFrame
    print('=' * 72)
    print(f'DIVERSIFICATION RECOMMENDATIONS  |  {len(recs)} picks (1 per sector)')
    print('=' * 72)
    print(f'Portfolio    : {USER_PORTFOLIO}')
    print('-' * 72)
    for _, row in recs.sort_values('sector').iterrows():
        flag  = '*' if row['is_medoid'] else ' '
        over  = '  ⚠ dcor>0.20' if row['avg_dcor_to_portfolio'] > 0.20 else ''
        print(f"  [{row['sector']:<30s}]  {flag}{row['stock']:<6s}  "
              f"dcor={row['avg_dcor_to_portfolio']:.4f}  "
              f"C{row['cluster']} (n={row['cluster_size']})  "
              f"{row['n_sector_candidates']} sector candidates{over}")
    print()
    print('* = cluster medoid  |  ⚠ = dcor above 0.20 (still within ceiling)')