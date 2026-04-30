import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
matplotlib.use('Agg')

TOP_N_PAIRS            = 15   # pairs to keep after filtering
MAX_PAIRS_PER_FOLLOWER =  2   # max incoming pairs per follower (stops cancellation)

# load and filtering pairs
raw_df = pd.read_csv('test.csv')

pairs_df = raw_df.rename(columns={
    'ticker_i':    'leader',
    'ticker_j':    'follower',
    'pearson_corr':'train_corr',
    'dcor':        'dcor_score',
    'q_value':     'pval',
}).copy()

# Sort by pearson_corr
pairs_df['abs_pearson'] = pairs_df['train_corr'].abs()
pairs_df = pairs_df.sort_values('abs_pearson', ascending=False).reset_index(drop=True)

selected, follower_counts = [], {}
for _, row in pairs_df.iterrows():
    if len(selected) >= TOP_N_PAIRS:
        break
    f = row['follower']
    if follower_counts.get(f, 0) < MAX_PAIRS_PER_FOLLOWER:
        selected.append(row)
        follower_counts[f] = follower_counts.get(f, 0) + 1

pairs_df = pd.DataFrame(selected).drop(columns=['abs_pearson']).reset_index(drop=True)
pairs_df['beta'] = pairs_df['train_corr']  # placeholder, replaced by calibrate_betas()

# stocks universe
all_stocks = sorted(set(pairs_df['leader'].tolist() + pairs_df['follower'].tolist()))
n_stocks   = len(all_stocks)
idx_map    = {s: i for i, s in enumerate(all_stocks)}

# helepr functions
def load_real_prices(tickers, start='2015-01-01', end='2025-01-01'):
    data = yf.download(tickers, start=start, end=end,
                       auto_adjust=True, progress=False)['Close']
    available = [t for t in tickers if t in data.columns]
    missing   = [t for t in tickers if t not in data.columns]
    if missing:
        print(f"  Not found: {missing}")
    return data[available].dropna(how='all').ffill().dropna()


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def extract_params(returns):
    mu    = returns.mean() * 252
    sigma = returns.std()  * np.sqrt(252)
    C = returns.corr().values.copy()
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-8)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(C))
    C = C / d[:, None] / d[None, :]
    L = np.linalg.cholesky(C)
    return {'mu': mu, 'sigma': sigma, 'L': L}


def calibrate_betas(pairs_df, train_ret):
    """OLS regression slope on training data"""
    betas = []
    for _, row in pairs_df.iterrows():
        lag      = int(row['lag'])
        leader   = row['leader']
        follower = row['follower']
        if leader not in train_ret.columns or follower not in train_ret.columns:
            betas.append(0.0)
            continue
        rl = train_ret[leader].iloc[:-lag].values
        rf = train_ret[follower].iloc[lag:].values
        slope, _, _, _, _ = stats.linregress(rl, rf)
        betas.append(slope)
    result = pairs_df.copy()
    result['beta'] = betas
    return result


def build_var(pairs_df, max_lag=5):
    A = {k: np.zeros((n_stocks, n_stocks)) for k in range(1, max_lag + 1)}
    for _, row in pairs_df.iterrows():
        lag = int(row['lag'])
        if lag <= max_lag:
            fi = idx_map[row['follower']]
            li = idx_map[row['leader']]
            if abs(row['beta']) > abs(A[lag][fi, li]):
                A[lag][fi, li] = row['beta']
    return A


def monte_carlo(params, var_A, S0, n_steps, n_sims=500):
    mu_v    = params['mu'].values
    sig_v   = params['sigma'].values
    L       = params['L']
    dt      = 1 / 252
    max_lag = max(var_A.keys())
    paths   = np.zeros((n_sims, n_steps + 1, n_stocks))
    paths[:, 0, :] = S0
    for sim in range(n_sims):
        ret_hist = np.zeros((max_lag + n_steps, n_stocks))
        for t in range(n_steps):
            z_c   = L @ np.random.normal(0, 1, n_stocks)
            var_c = sum(Ak @ ret_hist[max_lag + t - lag]
                        for lag, Ak in var_A.items()
                        if max_lag + t - lag >= 0)
            r_t                   = (mu_v - 0.5*sig_v**2)*dt + sig_v*np.sqrt(dt)*z_c + var_c
            ret_hist[max_lag + t] = r_t
            paths[sim, t+1, :]    = paths[sim, t, :] * np.exp(r_t)
    return paths


def percentile_bands(paths):
    return {k: np.percentile(paths, p, axis=0)
            for k, p in [('p5',5),('p16',16),('p50',50),('p84',84),('p95',95)]}


def plot_cones(train_p, test_p, bands_w, bands_wo, focus, title):
    BG         = '#ffffff'
    PANEL_BG   = '#f7f7f7'
    SPINE      = '#cccccc'
    TICK       = '#666666'
    TRAIN_CLR  = '#999999'
    SPLIT_CLR  = '#333333'
    CONE_CLR   = '#1a56db'
    ACTUAL_CLR = '#e02020'
    ANNOT_BG   = '#ffffff'
    ANNOT_BD   = '#cccccc'
    PAIRS_CLR  = '#e02020'
    TEXT_CLR   = '#111111'

    focus = [s for s in focus if s in idx_map]
    n     = len(focus)
    if n == 0:
        return

    fig = plt.figure(figsize=(18, 4.5 * n), facecolor=BG)
    fig.suptitle(title, color=TEXT_CLR, fontsize=13,
                 fontweight='bold', fontfamily='monospace', y=0.99)

    tl, tel = len(train_p), len(test_p)

    for ri, stock in enumerate(focus):
        si = idx_map[stock]
        for ci, (bands, lbl) in enumerate([
            (bands_wo, 'WITHOUT Lead-Lag'),
            (bands_w,  'WITH Lead-Lag')
        ]):
            ax = fig.add_subplot(n, 2, ri*2 + ci + 1)
            ax.set_facecolor(PANEL_BG)
            for sp in ax.spines.values():
                sp.set_color(SPINE)

            x_tr = np.arange(tl)
            x_te = np.arange(tl, tl + tel)
            x_bn = np.arange(tl, tl + bands['p50'].shape[0])
            T    = min(tel, bands['p50'].shape[0])

            ax.fill_between(x_bn, bands['p5'][:,si],  bands['p95'][:,si],
                            alpha=0.10, color=CONE_CLR, label='2σ band')
            ax.fill_between(x_bn, bands['p16'][:,si], bands['p84'][:,si],
                            alpha=0.22, color=CONE_CLR, label='1σ band')
            ax.plot(x_bn, bands['p50'][:,si],
                    color=CONE_CLR, lw=1.8, label='MC Median', zorder=3)
            ax.plot(x_tr, train_p[stock].values,
                    color=TRAIN_CLR, lw=1.2, label='Train (real)', zorder=2)
            ax.plot(x_te[:T], test_p[stock].values[:T],
                    color=ACTUAL_CLR, lw=2.2, label='Actual (real)', zorder=5)

            ax.axvline(x=tl, color=SPLIT_CLR, ls='--', alpha=0.5, lw=1)
            ax.text(tl+5, ax.get_ylim()[0], 'TEST →',
                    color=SPLIT_CLR, fontsize=7, fontfamily='monospace', alpha=0.6)

            mid = T // 2
            w   = (bands['p84'][mid,si] - bands['p16'][mid,si]) / bands['p50'][mid,si] * 100
            ax.text(0.98, 0.04, f'1\u03c3 width: \u00b1{w/2:.1f}%',
                    transform=ax.transAxes, ha='right', va='bottom',
                    color=CONE_CLR, fontsize=8, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=ANNOT_BG,
                              edgecolor=ANNOT_BD, alpha=0.9))

            actual_vals = test_p[stock].values[:T]
            in_band     = np.mean(
                (actual_vals >= bands['p16'][:T,si]) &
                (actual_vals <= bands['p84'][:T,si])
            )
            ax.text(0.98, 0.13, f'1\u03c3 coverage: {in_band:.1%}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    color=ACTUAL_CLR, fontsize=8, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=ANNOT_BG,
                              edgecolor=ANNOT_BD, alpha=0.9))

            col = '#1a56db' if 'WITH' in lbl else '#333333'
            ax.set_title(f"{stock}  \u2014  {lbl}", color=col,
                         fontsize=9, fontfamily='monospace', pad=6)
            ax.tick_params(colors=TICK, labelsize=7)

            related = pairs_df[
                (pairs_df['leader']==stock) | (pairs_df['follower']==stock)
            ]
            if not related.empty and ci == 1:
                pt = ', '.join([
                    f"{r['leader']}\u2192{r['follower']} lag{int(r['lag'])}d \u03b2={r['beta']:+.3f}"
                    for _, r in related.iterrows()
                ])
                ax.text(0.02, 0.97, f'Pairs: {pt}',
                        transform=ax.transAxes, ha='left', va='top',
                        color=PAIRS_CLR, fontsize=7, fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=ANNOT_BG,
                                  edgecolor=ANNOT_BD, alpha=0.9))

            if ri == 0 and ci == 0:
                ax.legend(loc='upper left', fontsize=7, framealpha=0.9,
                          facecolor=ANNOT_BG, edgecolor=ANNOT_BD, labelcolor=TEXT_CLR)

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig('cone_comparison_REAL.png', dpi=140,
                bbox_inches='tight', facecolor=BG)
    print("  Saved -> cone_comparison_REAL.png")

# main pipeline
print("Downloading data")
prices_all = load_real_prices(all_stocks, start='2015-01-01', end='2025-01-01')

all_stocks = list(prices_all.columns)
n_stocks   = len(all_stocks)
idx_map    = {s: i for i, s in enumerate(all_stocks)}

pairs_df = pairs_df[
    pairs_df['leader'].isin(all_stocks) & pairs_df['follower'].isin(all_stocks)
].reset_index(drop=True)

split_date   = pd.Timestamp('2020-01-01')
train_prices = prices_all[prices_all.index < split_date]
test_prices  = prices_all[prices_all.index >= split_date]

train_ret = log_returns(train_prices)
test_ret  = log_returns(test_prices)

params   = extract_params(train_ret)
pairs_df = calibrate_betas(pairs_df, train_ret)
var_A    = build_var(pairs_df, max_lag=5)

print("Running")
S0         = test_prices.iloc[0].values
paths_with = monte_carlo(params, var_A, S0, len(test_prices), n_sims=500)
bands_with = percentile_bands(paths_with)

empty_var = {k: np.zeros((n_stocks, n_stocks)) for k in range(1, 6)}
paths_wo  = monte_carlo(params, empty_var, S0, len(test_prices), n_sims=500)
bands_wo  = percentile_bands(paths_wo)

focus_stocks = sorted(set(pairs_df['leader'].tolist() + pairs_df['follower'].tolist()))
plot_cones(train_prices, test_prices, bands_with, bands_wo,
           focus_stocks,
           'Cone Comparison: WITH vs WITHOUT Lead-Lag')
print("Complete")