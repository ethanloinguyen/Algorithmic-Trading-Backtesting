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

print('All imports successful ✓')
import sklearn
print(f'  sklearn={sklearn.__version__}  numpy={np.__version__}')


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


print('KMedoidsPAM defined ✓')


# =============================================================================
# 2 · Configuration  ← Edit these values to match your project
# =============================================================================

GCP_PROJECT    = 'capstone-487001'
BQ_TABLE       = f'{GCP_PROJECT}.output_results.final_network'
COL_STOCK_A    = 'ticker_i'
COL_STOCK_B    = 'ticker_j'
COL_DCOR       = 'mean_dcor'
USER_PORTFOLIO = ['AAPL', 'MSFT', 'GOOGL']
DCOR_THRESHOLD = 0.25

NUMERIC_FEATURES = [
    'mean_dcor', 'variance_dcor', 'best_lag', 'frequency', 'half_life',
    'sharpness', 'predicted_sharpe', 'signal_strength',
    'oos_sharpe_net', 'oos_dcor', 'centrality_i',
]
USE_SECTOR    = True
SECTOR_WEIGHT = 1.0
CORRELATION_DROP_THRESHOLD = 0.85
PCA_COMPONENTS = None

# Clustering
MIN_CLUSTER_SIZE     = 8
K_MIN                = 3
K_MAX                = None   # auto: pool_size // MIN_CLUSTER_SIZE
N_INIT               = 25
MAX_CLUSTER_FRACTION = 0.25

# Objective-driven k selection
ELBOW_SENSITIVITY  = 0.45
MIN_SECTORS_TARGET = 6

# Set to an integer to pin k manually; None = objective selection
N_CLUSTERS = None

print('Config loaded ✓')
print(f'  Portfolio              : {USER_PORTFOLIO}')
print(f'  DCOR_THRESHOLD         : {DCOR_THRESHOLD}')
print(f'  SECTOR_WEIGHT          : {SECTOR_WEIGHT}')
print(f'  MIN_CLUSTER_SIZE       : {MIN_CLUSTER_SIZE}')
print(f'  N_INIT                 : {N_INIT}')
print(f'  ELBOW_SENSITIVITY      : {ELBOW_SENSITIVITY}')
print(f'  MIN_SECTORS_TARGET     : {MIN_SECTORS_TARGET}')
print(f'  MAX_CLUSTER_FRACTION   : {MAX_CLUSTER_FRACTION}')
print(f'  Output                 : 1 stock per sector (global best-per-sector)')
print(f'  N_CLUSTERS override    : {N_CLUSTERS}')


# =============================================================================
# 3 · BigQuery — Candidate Pool
# =============================================================================

client = bigquery.Client(project=GCP_PROJECT)
print('BigQuery client initialised ✓')


def build_candidate_pool(portfolio, bq_table, dcor_threshold):
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


candidate_df     = build_candidate_pool(USER_PORTFOLIO, BQ_TABLE, DCOR_THRESHOLD)
CANDIDATE_STOCKS = candidate_df['candidate_stock'].tolist()
if K_MAX is None:
    K_MAX = len(CANDIDATE_STOCKS) // MIN_CLUSTER_SIZE
    print(f'Auto K_MAX = {K_MAX}')
print(candidate_df.head(10).to_string())


# =============================================================================
# 4 · Feature Vectors (Bidirectional Aggregation)
# =============================================================================

def fetch_feature_rows_bidirectional(stocks, bq_table):
    tickers_sql = ', '.join(f"'{t}'" for t in stocks)
    feat_i = ', '.join(NUMERIC_FEATURES)
    sector_i = ', sector_i' if USE_SECTOR else ''
    feat_j_parts = ['centrality_j AS centrality_i' if f == 'centrality_i' else f
                    for f in NUMERIC_FEATURES]
    feat_j = ', '.join(feat_j_parts)
    sector_j = ', sector_j AS sector_i' if USE_SECTOR else ''
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


def build_feature_matrix(feature_rows, candidate_stocks):
    numeric_agg = feature_rows.groupby('stock')[NUMERIC_FEATURES].median().reindex(candidate_stocks)
    numeric_df = pd.DataFrame(
        SimpleImputer(strategy='median').fit_transform(numeric_agg),
        index=candidate_stocks, columns=NUMERIC_FEATURES
    )
    if USE_SECTOR:
        sector_mode = (
            feature_rows.groupby('stock')['sector_i']
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown')
            .reindex(candidate_stocks, fill_value='Unknown')
        )
        sector_dummies = pd.get_dummies(sector_mode, prefix='sector') * SECTOR_WEIGHT
        print(f'Sectors: {sorted(sector_mode.unique())}')
        out = pd.concat([numeric_df, sector_dummies], axis=1)
    else:
        out = numeric_df
    print(f'Feature matrix: {out.shape[0]} x {out.shape[1]}')
    return out


feature_rows   = fetch_feature_rows_bidirectional(CANDIDATE_STOCKS, BQ_TABLE)
feature_matrix = build_feature_matrix(feature_rows, CANDIDATE_STOCKS)

sector_map = feature_rows.groupby('stock')['sector_i'].agg(
    lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown')
print(f'sector_map built for {len(sector_map)} stocks')


# =============================================================================
# 5 · Feature Selection & Scaling
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
        for col in to_drop:
            partners = upper.index[upper[col] > threshold].tolist()
            print(f'  {col} corr>{threshold} with {partners}')
    else:
        print(f'No features dropped (all correlations <= {threshold})')
    print(f'Kept numeric ({len(kept)}): {kept}')
    return df[kept + sector_cols], kept


feature_matrix_sel, kept_numeric = drop_correlated_features(feature_matrix, CORRELATION_DROP_THRESHOLD)

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(feature_matrix_sel.values)

if PCA_COMPONENTS is not None:
    pca_t    = PCA(n_components=min(PCA_COMPONENTS, X_scaled.shape[1]-1), whiten=True, random_state=42)
    X_scaled = pca_t.fit_transform(X_scaled)
    print(f'PCA: {X_scaled.shape[1]} components, {pca_t.explained_variance_ratio_.sum():.1%} variance')

X_df = pd.DataFrame(X_scaled, index=feature_matrix_sel.index)
D    = pairwise_distances(X_scaled, metric='euclidean').astype(np.float64)
D   /= D.max()
upper_tri = D[np.triu_indices(len(D), k=1)]

# Variance decomposition
if PCA_COMPONENTS is None:
    sector_cols_sel  = [c for c in feature_matrix_sel.columns if c.startswith('sector_')]
    numeric_cols_sel = [c for c in feature_matrix_sel.columns if not c.startswith('sector_')]
    if sector_cols_sel:
        si = [list(feature_matrix_sel.columns).index(c) for c in sector_cols_sel]
        ni = [list(feature_matrix_sel.columns).index(c) for c in numeric_cols_sel]
        sv = X_scaled[:, si].var(axis=0).sum()
        nv = X_scaled[:, ni].var(axis=0).sum()
        print(f'\nVariance decomposition: numeric={nv/(nv+sv):.1%}  sector={sv/(nv+sv):.1%}')
        print(f'  Target: sector 20-40%.  If >40%, lower SECTOR_WEIGHT further.')

print(f'\nDistance std/mean ratio: {upper_tri.std()/upper_tri.mean():.3f}')


# =============================================================================
# 6 · K-Medoids Sweep
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


if K_MAX is None:
    K_MAX = max(K_MIN + 1, len(CANDIDATE_STOCKS) // MIN_CLUSTER_SIZE)
    print(f'Auto K_MAX = {K_MAX}  (pool={len(CANDIDATE_STOCKS)} / min_size={MIN_CLUSTER_SIZE})')

sweep = run_full_sweep(D, K_MIN, K_MAX, N_INIT, MIN_CLUSTER_SIZE, MAX_CLUSTER_FRACTION)
print('\n⚠ = largest cluster exceeds', f'{MAX_CLUSTER_FRACTION:.0%}',
      'of pool — picks from that cluster carry less confidence')


# =============================================================================
# 7 · Objective-Driven k Selection
# =============================================================================

def build_cluster_df_for_k(k):
    labels  = sweep[k]['labels']
    medoids = sweep[k]['medoids']
    cdf = (
        pd.DataFrame({'stock': CANDIDATE_STOCKS, 'cluster': labels + 1})
        .merge(candidate_df[['candidate_stock', 'avg_dcor_to_portfolio', 'min_dcor_to_portfolio']],
               left_on='stock', right_on='candidate_stock', how='left')
    )
    stock_feats = feature_matrix[kept_numeric].copy()
    cdf = cdf.merge(stock_feats, left_on='stock', right_index=True, how='left')
    if USE_SECTOR:
        cdf['sector'] = cdf['stock'].map(sector_map)
    cdf['is_medoid'] = cdf['stock'].isin([CANDIDATE_STOCKS[i] for i in medoids])
    return cdf


def pick_best_per_sector(cluster_df):
    dist_df = pd.DataFrame(D, index=CANDIDATE_STOCKS, columns=CANDIDATE_STOCKS)
    intra_series = {}
    for cid, group in cluster_df.groupby('cluster'):
        members  = group['stock'].tolist()
        sub_dist = dist_df.loc[members, members]
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
        sector_stocks['avg_dcor_to_portfolio'] = sector_stocks['avg_dcor_to_portfolio'].fillna(DCOR_THRESHOLD)
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
    print(f'Stage A — Elbow detection (valid k only, negative drops clipped):')
    print(f'  Valid k range            : {min(ks)}–{max(ks)}')
    print(f'  Max-acceleration method  : k={e_accel}')
    print(f'  Sensitivity ({sensitivity}) method: k={e_sens}')
    print(f'  Floor (conservative)     : k={elbow_k}')
    return elbow_k, ks, inertias, drops, accels


def select_k_by_sector_diversity(sweep, elbow_k, min_sectors_target):
    valid_ks = [k for k in sorted(sweep) if sweep[k]['valid'] and k >= elbow_k]
    if not valid_ks:
        valid_ks = [k for k in sorted(sweep) if sweep[k]['valid']]
        print('  WARNING: No valid k >= elbow floor — using all valid k')
    print(f'\nStage B — Sector diversity across valid k >= {elbow_k}:')
    print(f'{"k":>4}  {"sectors":>8}  {"picks":>6}  {"silhouette":>11}  {"oversized":>10}  sector_list')
    print('-' * 90)
    div = {}
    for k in valid_ks:
        cdf  = build_cluster_df_for_k(k)
        recs = pick_best_per_sector(cdf)
        secs = recs['sector'].tolist()
        over = sweep[k]['oversized']
        div[k] = {'n_sectors': len(secs), 'sectors': secs,
                  'n_picks': len(recs), 'sil': sweep[k]['sil'],
                  'oversized': over}
        marker = ' <-- meets target' if len(secs) >= min_sectors_target else ''
        owarn  = '  ⚠' if over else '   '
        print(f'{k:>4}  {len(secs):>8}  {len(recs):>6}  '
              f'{sweep[k]["sil"]:>+11.4f}  {owarn:>10}  {sorted(secs)}{marker}')
    best_k = max(div, key=lambda k: (div[k]['n_sectors'], div[k]['sil']))
    r = div[best_k]
    print(f'\n  -> Selected k={best_k}  '
          f'({r["n_sectors"]} sectors, {r["n_picks"]} picks, sil={r["sil"]:.4f})')
    if r['n_sectors'] < min_sectors_target:
        print(f'  NOTE: Best sectors ({r["n_sectors"]}) < MIN_SECTORS_TARGET ({min_sectors_target})')
    if r['oversized']:
        print(f'  ⚠  k={best_k} has a cluster exceeding {MAX_CLUSTER_FRACTION:.0%} of pool.')
    return best_k, div


# Run k selection
if N_CLUSTERS is None:
    print('=' * 60)
    elbow_k, ks_all, inertias_all, drops_all, accels_all = detect_elbow(sweep, ELBOW_SENSITIVITY)
    print('=' * 60)
    N_CLUSTERS, diversity_results = select_k_by_sector_diversity(
        sweep, elbow_k, MIN_SECTORS_TARGET
    )
else:
    print(f'N_CLUSTERS manually pinned to {N_CLUSTERS} — skipping objective selection')
    elbow_k      = N_CLUSTERS
    ks_all       = [k for k in sorted(sweep) if sweep[k]['valid']]
    inertias_all = [sweep[k]['inertia'] for k in ks_all]
    drops_all    = [max(0.0, inertias_all[i]-inertias_all[i+1]) for i in range(len(inertias_all)-1)]
    accels_all   = [drops_all[i]-drops_all[i+1] for i in range(len(drops_all)-1)]
    diversity_results = {}

print(f'\n✅ Final k = {N_CLUSTERS}')


# (Elbow and sector-diversity diagnostic plots removed — outputs are CSV + star graph only)


# =============================================================================
# 8 · Assign Clusters, Select Representatives & Plot
# =============================================================================

cluster_df = build_cluster_df_for_k(N_CLUSTERS)
sizes = cluster_df['cluster'].value_counts().sort_index()
print(f'k={N_CLUSTERS}  sil={sweep[N_CLUSTERS]["sil"]:.4f}  bal_cv={sweep[N_CLUSTERS]["bal_cv"]:.3f}')
print(f'Min={sizes.min()}  Max={sizes.max()}  Mean={sizes.mean():.1f}  Std={sizes.std():.1f}')
print(f'\nCluster sizes:')
print(sizes.to_string())
print(f'\nMedoids: {[CANDIDATE_STOCKS[i] for i in sweep[N_CLUSTERS]["medoids"]]}')

if sweep[N_CLUSTERS]['oversized']:
    print(f'  ⚠  Largest cluster ({sizes.max()}) exceeds '
          f'{MAX_CLUSTER_FRACTION:.0%} of pool ({len(CANDIDATE_STOCKS)} stocks)')

recommendations = pick_best_per_sector(cluster_df)

print(f'\n{len(recommendations)} recommendations — one per sector (sorted by dcor ascending):')
print(recommendations[[
    'sector', 'stock', 'cluster', 'is_medoid',
    'avg_dcor_to_portfolio', 'mean_intra_dist',
    'n_sector_candidates', 'cluster_size'
]].to_string(index=False))

# Sector uniqueness check
print('Sector uniqueness check:')
sec_counts = recommendations['sector'].value_counts()
all_unique  = all(c == 1 for c in sec_counts)
for sec, cnt in sec_counts.items():
    print(f'  {sec:<30s}: {cnt}  -> {"OK" if cnt == 1 else "FAIL"}')
print(f'\nAll sectors unique: {"✅" if all_unique else "❌"}')

dcor_vals = recommendations['avg_dcor_to_portfolio'].dropna()
print(f'\ndcor range of picks: [{dcor_vals.min():.4f}, {dcor_vals.max():.4f}]')
print(f'dcor mean of picks : {dcor_vals.mean():.4f}')
elevated = recommendations[recommendations['avg_dcor_to_portfolio'] > 0.20]
if not elevated.empty:
    print(f'\nPicks with dcor > 0.20:')
    for _, row in elevated.iterrows():
        print(f'  {row["stock"]:<6s}  [{row["sector"]}]  dcor={row["avg_dcor_to_portfolio"]:.4f}')

# Quality check
print('=' * 60)
print('CLUSTERING QUALITY CHECK')
print('=' * 60)
failures = []

print('\n[1] One stock per sector (global):')
sec_counts = recommendations['sector'].value_counts()
for sec, cnt in sec_counts.items():
    status = 'OK' if cnt == 1 else f'FAIL — {cnt} stocks'
    print(f'  {sec:<30s}: {cnt}  -> {status}')
    if cnt != 1: failures.append(f'{sec} has {cnt} stocks (expected 1)')

n_sectors = len(recommendations)
print(f'\n[2] Minimum sector coverage (target >= {MIN_SECTORS_TARGET}):')
status = 'OK' if n_sectors >= MIN_SECTORS_TARGET else f'FAIL — only {n_sectors}'
print(f'  {n_sectors} sectors represented  -> {status}')
if n_sectors < MIN_SECTORS_TARGET:
    failures.append(f'Only {n_sectors} sectors < target {MIN_SECTORS_TARGET}')

print(f'\n[3] dcor ceiling ({DCOR_THRESHOLD}) for all picks:')
over_ceiling = recommendations[recommendations['avg_dcor_to_portfolio'] > DCOR_THRESHOLD]
if over_ceiling.empty:
    print(f'  All picks below dcor ceiling  -> OK')
else:
    for _, row in over_ceiling.iterrows():
        print(f'  FAIL — {row["stock"]} ({row["sector"]}) dcor={row["avg_dcor_to_portfolio"]:.4f}')
        failures.append(f'{row["stock"]} exceeds dcor ceiling')

print(f'\n[4] Cluster size balance (max fraction {MAX_CLUSTER_FRACTION:.0%}):')
if sweep[N_CLUSTERS]['oversized']:
    sizes_vc = cluster_df['cluster'].value_counts()
    big = sizes_vc[sizes_vc > len(CANDIDATE_STOCKS) * MAX_CLUSTER_FRACTION]
    for cid, sz in big.items():
        print(f'  ⚠  Cluster {cid}: {sz} stocks ({sz/len(CANDIDATE_STOCKS):.1%} of pool) — WARNING only')
else:
    print(f'  All clusters within {MAX_CLUSTER_FRACTION:.0%} of pool  -> OK')

print('\n' + '=' * 60)
if failures:
    print('STATUS: ❌ QUALITY CHECK FAILED')
    for f in failures: print(f'  - {f}')
    raise SystemExit('Quality check failed — resolve before exporting')
else:
    print('STATUS: ✅ ALL CHECKS PASSED')
    print(f'  {n_sectors} sectors  |  {n_sectors} picks  |  k={N_CLUSTERS}  |  sil={sweep[N_CLUSTERS]["sil"]:.4f}')
print('=' * 60)


# =============================================================================
# 9 · Sector Star Graph
# t-SNE layout; each recommended stock shown as a ★ labelled with ticker+sector.
# All other candidate stocks shown as faint grey dots for context.
# =============================================================================

coords = TSNE(n_components=2, random_state=42,
              perplexity=min(30, len(CANDIDATE_STOCKS) - 1),
              init='pca', max_iter=1000).fit_transform(X_scaled)

rec_tickers = set(recommendations['stock'])
sectors_sorted = sorted(recommendations['sector'].unique())
sector_palette = {s: plt.cm.tab20(i / max(len(sectors_sorted) - 1, 1))
                  for i, s in enumerate(sectors_sorted)}

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

# Background: all non-recommended candidates as faint grey dots
for idx, ticker in enumerate(CANDIDATE_STOCKS):
    if ticker not in rec_tickers:
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   color='#cccccc', s=25, alpha=0.4, zorder=1)

# Foreground: one ★ per recommended stock, coloured by sector
for _, row in recommendations.iterrows():
    idx   = CANDIDATE_STOCKS.index(row['stock'])
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
    f'Sector Star Graph — k={N_CLUSTERS}  |  {len(recommendations)} recommended stocks  '
    f'(★ = sector pick,  · = candidate pool)',
    fontsize=13, fontweight='bold', pad=14
)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

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


# =============================================================================
# 10 · Export
# =============================================================================

recommendations.to_csv('recommended_stocks.csv', index=False)

sil_val = sweep[N_CLUSTERS]['sil']
rating  = ('STRONG' if sil_val>=0.5 else 'MODERATE' if sil_val>=0.25
           else 'WEAK' if sil_val>=0 else 'POOR')

print('=' * 72)
print(f'DIVERSIFICATION RECOMMENDATIONS  |  {len(recommendations)} picks (1 per sector)')
print('=' * 72)
print(f'Portfolio    : {USER_PORTFOLIO}')
print(f'k selected   : elbow floor ({elbow_k}) + sector diversity')
print(f'Silhouette   : {sil_val:.4f}  ({rating})')
print(f'Balance CV   : {sweep[N_CLUSTERS]["bal_cv"]:.3f}')
print(f'Sectors      : {len(recommendations)}')
print('-' * 72)
for _, row in recommendations.sort_values('sector').iterrows():
    flag  = '*' if row['is_medoid'] else ' '
    over  = '  ⚠ dcor>0.20' if row['avg_dcor_to_portfolio'] > 0.20 else ''
    print(f"  [{row['sector']:<30s}]  {flag}{row['stock']:<6s}  "
          f"dcor={row['avg_dcor_to_portfolio']:.4f}  "
          f"C{row['cluster']} (n={row['cluster_size']})  "
          f"{row['n_sector_candidates']} sector candidates{over}")
print()
print('-' * 72)
print('* = cluster medoid  |  ⚠ = dcor above 0.20 (still within ceiling)')
print()
print('TUNING GUIDE:')
print('  More sectors     -> lower DCOR_THRESHOLD or raise MIN_SECTORS_TARGET')
print('  Elbow too low    -> raise ELBOW_SENSITIVITY (currently', ELBOW_SENSITIVITY, ')')
print('  Elbow too high   -> lower ELBOW_SENSITIVITY')
print('  Oversized cluster -> raise MIN_CLUSTER_SIZE (currently', MIN_CLUSTER_SIZE, ')')
print('  Override k        -> set N_CLUSTERS = <integer>')
print('Saved: recommended_stocks.csv, sector_star_graph.png')