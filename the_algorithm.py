"""
STREAMLINED LEAD-LAG DETECTION
Focus: Accurate pairwise relationships with robust validation
Removed: Hermitian clustering (not needed for pairwise analysis)
Added: Data preprocessing, partial correlation, FDR, temporal validation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from dcor import distance_correlation
from scipy.stats import false_discovery_control, pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

np.random.seed(42)

# Configuration
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
MAX_LAG = 5
MIN_TRADING_DAYS = 250  # At least 1 year of data
MIN_AVG_VOLUME = 1_000_000  # Filter low-volume stocks
MAX_MISSING_PCT = 0.05  # Max 5% missing data
OUTLIER_THRESHOLD = 5.0  # Remove returns beyond ±5 std devs
MIN_PERSISTENCE = 0.5  # Relationship must appear in ≥50% of time windows
FDR_ALPHA = 0.05  # False discovery rate
N_BOOTSTRAP = 100  # For p-value computation
SYNC_THRESHOLD = 0.3  # Threshold for synchronous filtering

# ============================================================================
# SECTION 1: ENHANCED DATA PREPROCESSING
# ============================================================================
def get_russell3000_tickers(n_stocks: int = 2000) -> list[str]:
    """
    Get list of top N Russell 3000 tickers using the IWV ETF as a proxy.
    
    Args:
        n_stocks (int): Number of top stocks to retrieve (default 2000).
        
    Returns:
        List[str]: List of valid tickers compatible with yfinance.
    """
    # URL to the holdings of iShares Russell 3000 ETF (IWV)
    # This CSV is sorted by weight (market cap) by default.
    url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
    
    try:
        # Fetch the CSV content
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse CSV. iShares CSVs typically have ~9 rows of metadata before the header.
        csv_content = response.content.decode('utf-8')
        
        # Find the header row index programmatically or default to 9
        # (Looking for the line starting with "Ticker")
        skip_rows = 9
        
        df = pd.read_csv(io.StringIO(csv_content), skiprows=skip_rows)
        
        # Ensure we have the correct columns and clean data
        if 'Ticker' not in df.columns:
            # Fallback for different CSV formatting
            df = pd.read_csv(io.StringIO(csv_content), header=0)
            
        # Filter out non-equity assets (Cash, Derivatives, etc.) if labeled
        if 'Asset Class' in df.columns:
            df = df[df['Asset Class'] == 'Equity']
            
        # Get the ticker column
        tickers = df['Ticker'].dropna().astype(str).tolist()
        
        # CLEANING: yfinance uses '-' for classes (e.g. BRK-B), while many sources use '.' (BRK.B)
        tickers = [t.replace('.', '-') for t in tickers]
        
        # Return the top N stocks
        return tickers[:n_stocks]

    except Exception as e:
        print(f"Error fetching Russell 3000 list: {e}")
        return []


def get_sp500_tickers(n_stocks: int = 50) -> List[str]:
    """Get list of S&P 500 tickers."""
    sp500_top_n = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'UNH',
        'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV', 'KO',
        'PEP', 'AVGO', 'COST', 'TMO', 'MCD', 'CSCO', 'ABT', 'ACN', 'DHR', 'LIN',
        'ADBE', 'NKE', 'TXN', 'PM', 'NEE', 'DIS', 'CMCSA', 'VZ', 'INTC', 'CRM',
        'AMD', 'NFLX', 'QCOM', 'UNP', 'IBM', 'T', 'RTX', 'HON', 'AMGN', 'BA'
    ]
    return sp500_top_n[:n_stocks]


def compute_data_quality_metrics(stock_data: pd.DataFrame) -> Dict:
    """
    Compute quality metrics for a stock's data.
    
    Returns:
    - n_trading_days: Number of trading days
    - missing_pct: Percentage of missing data
    - avg_volume: Average daily volume
    - volatility: Annualized volatility
    - has_outliers: Whether extreme outliers exist
    """
    returns = np.log(stock_data['close'] / stock_data['close'].shift(1)).dropna()
    
    metrics = {
        'n_trading_days': len(stock_data),
        'missing_pct': stock_data['close'].isna().sum() / len(stock_data),
        'avg_volume': stock_data['volume'].mean(),
        'volatility': returns.std() * np.sqrt(252),
        'has_outliers': (np.abs(returns) > OUTLIER_THRESHOLD * returns.std()).any()
    }
    
    return metrics


def preprocess_stock_data(stock_data: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, bool]:
    """
    Preprocess and validate stock data.
    
    Returns:
    - Cleaned DataFrame
    - Boolean indicating if stock passes quality checks
    """
    # Remove duplicates
    stock_data = stock_data[~stock_data.index.duplicated(keep='first')]
    
    # Compute quality metrics
    metrics = compute_data_quality_metrics(stock_data)
    
    # Quality checks
    passes_quality = True
    reasons = []
    
    if metrics['n_trading_days'] < MIN_TRADING_DAYS:
        passes_quality = False
        reasons.append(f"insufficient data ({metrics['n_trading_days']} days)")
    
    if metrics['missing_pct'] > MAX_MISSING_PCT:
        passes_quality = False
        reasons.append(f"too much missing data ({metrics['missing_pct']:.1%})")
    
    if metrics['avg_volume'] < MIN_AVG_VOLUME:
        passes_quality = False
        reasons.append(f"low volume ({metrics['avg_volume']:,.0f})")
    
    if not passes_quality:
        print(f"  ✗ {ticker}: {', '.join(reasons)}")
        return None, False
    
    # Forward-fill missing values (max 5 days)
    stock_data['close'] = stock_data['close'].fillna(method='ffill', limit=5)
    stock_data['volume'] = stock_data['volume'].fillna(method='ffill', limit=5)
    
    # Compute returns
    stock_data['returns'] = np.log(stock_data['close'] / stock_data['close'].shift(1))
    
    # Remove outliers (winsorize extreme returns)
    returns_std = stock_data['returns'].std()
    stock_data['returns'] = stock_data['returns'].clip(
        lower=-OUTLIER_THRESHOLD * returns_std,
        upper=OUTLIER_THRESHOLD * returns_std
    )
    
    # Drop first row (NaN return)
    stock_data = stock_data.dropna(subset=['returns'])
    
    print(f"  ✓ {ticker}: {len(stock_data)} days, "
          f"vol={metrics['avg_volume']:,.0f}, "
          f"σ={metrics['volatility']:.1%}")
    
    return stock_data, True


def load_and_preprocess_data(tickers: List[str], start_date: str, 
                             end_date: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load stock data with robust preprocessing and quality checks.
    """
    print(f"\n{'='*60}")
    print("SECTION 1: DATA LOADING & PREPROCESSING")
    print(f"{'='*60}\n")
    
    print(f"Quality filters:")
    print(f"  - Minimum trading days: {MIN_TRADING_DAYS}")
    print(f"  - Minimum avg volume: {MIN_AVG_VOLUME:,}")
    print(f"  - Maximum missing data: {MAX_MISSING_PCT:.1%}")
    print(f"  - Outlier threshold: ±{OUTLIER_THRESHOLD}σ")
    
    # Download data
    print(f"\nDownloading data for {len(tickers)} stocks...")
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        group_by='ticker',
        auto_adjust=True
    )
    
    # Process each stock
    print("\nProcessing and validating stocks:")
    processed_data = []
    successful_tickers = []
    
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                stock_data = data[['Close', 'Volume']].copy()
            else:
                stock_data = data[ticker][['Close', 'Volume']].copy()
            
            stock_data.columns = ['close', 'volume']
            stock_data['ticker'] = ticker
            
            # Preprocess and validate
            cleaned_data, passes = preprocess_stock_data(stock_data, ticker)
            
            if passes and cleaned_data is not None:
                processed_data.append(cleaned_data)
                successful_tickers.append(ticker)
                
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")
            continue
    
    if not processed_data:
        raise ValueError("No stocks passed quality checks!")
    
    returns_df = pd.concat(processed_data, axis=0)
    
    print(f"\n{'='*60}")
    print(f"Data loaded: {len(successful_tickers)}/{len(tickers)} stocks passed quality checks")
    print(f"Date range: {returns_df.index.min().date()} to {returns_df.index.max().date()}")
    print(f"{'='*60}\n")
    
    return returns_df, successful_tickers


# ============================================================================
# SECTION 2: PARTIAL CORRELATION LEAD-LAG COMPUTATION
# ============================================================================

def partial_distance_correlation(x: np.ndarray, y: np.ndarray, 
                                 control: np.ndarray) -> float:
    """
    Compute partial distance correlation: dcor(x, y | control)
    
    This removes the effect of 'control' variable from both x and y,
    then computes distance correlation on residuals.
    
    Critical for removing spurious lead-lag from:
    - Synchronous correlation × Auto-correlation
    """
    n = min(len(x), len(y), len(control))
    x, y, control = x[-n:], y[-n:], control[-n:]
    
    # Reshape for sklearn
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    control = control.reshape(-1, 1)
    
    # Residualize x ~ control
    model_x = LinearRegression()
    model_x.fit(control, x)
    residual_x = x - model_x.predict(control)
    
    # Residualize y ~ control
    model_y = LinearRegression()
    model_y.fit(control, y)
    residual_y = y - model_y.predict(control)
    
    # Distance correlation of residuals
    try:
        return distance_correlation(residual_x.ravel(), residual_y.ravel())
    except:
        return 0.0


def compute_lag_correlations_controlled(returns_i: np.ndarray, returns_j: np.ndarray,
                                        max_lag: int = 5) -> Dict[int, float]:
    """
    Compute distance correlation at each lag, with partial correlation control.
    
    For each lag:
    - Controls for synchronous correlation
    - Removes spurious lead-lag from auto-correlation
    
    Returns: {lag: partial_correlation_value}
    """
    n = min(len(returns_i), len(returns_j))
    returns_i = returns_i[-n:]
    returns_j = returns_j[-n:]
    
    lag_correlations = {}
    
    # Lag 0 (synchronous) - regular distance correlation
    try:
        lag_correlations[0] = distance_correlation(returns_i, returns_j)
    except:
        lag_correlations[0] = 0.0
    
    # Positive lags (i leads j) - with partial correlation
    for lag in range(1, max_lag + 1):
        if lag < n:
            try:
                # Partial correlation: control for i(t) when correlating i(t-lag) with j(t)
                # This removes spurious lead-lag from auto-correlation
                i_lagged = returns_i[:-lag]
                j_current = returns_j[lag:]
                i_current = returns_i[lag:]  # Control variable
                
                partial_corr = partial_distance_correlation(
                    i_lagged, j_current, control=i_current
                )
                lag_correlations[lag] = partial_corr
            except:
                lag_correlations[lag] = 0.0
    
    # Negative lags (j leads i) - with partial correlation
    for lag in range(1, max_lag + 1):
        if lag < n:
            try:
                j_lagged = returns_j[:-lag]
                i_current = returns_i[lag:]
                j_current = returns_j[lag:]  # Control variable
                
                partial_corr = partial_distance_correlation(
                    j_lagged, i_current, control=j_current
                )
                lag_correlations[-lag] = partial_corr
            except:
                lag_correlations[-lag] = 0.0
    
    return lag_correlations


def is_synchronous_relationship(lag_corrs: Dict[int, float], 
                                sync_threshold: float = SYNC_THRESHOLD) -> bool:
    """
    Check if relationship is primarily synchronous (lag=0 dominant).
    
    After partial correlation control, this should catch remaining
    synchronous relationships that don't have true lead-lag.
    """
    sync_corr = abs(lag_corrs.get(0, 0))
    
    lagged_corrs = [abs(v) for k, v in lag_corrs.items() if k != 0]
    max_lagged = max(lagged_corrs) if lagged_corrs else 0
    
    # Sync much stronger than any lag
    if sync_corr > max_lagged + sync_threshold:
        return True
    
    # Sync is dominant (>1.5x strongest lag)
    if max_lagged > 0 and sync_corr / max_lagged > 1.5:
        return True
    
    return False


def compute_lead_lag_with_partial_correlation(returns_i: np.ndarray, returns_j: np.ndarray,
                                               max_lag: int = 5) -> Tuple[float, int, Dict, bool]:
    """
    Compute lead-lag metric with partial correlation control.
    
    This is the CORRECTED version that removes spurious lead-lag.
    
    Returns:
    - lead_lag_metric: Signed metric (-1 to +1)
    - optimal_lag: Lag with strongest correlation
    - lag_correlations: All correlations at each lag
    - is_sync: Whether relationship is primarily synchronous
    """
    # Compute partial correlations at each lag
    lag_corrs = compute_lag_correlations_controlled(returns_i, returns_j, max_lag)
    
    # Check if synchronous
    is_sync = is_synchronous_relationship(lag_corrs)
    
    # Find optimal lag (excluding lag=0)
    non_sync_lags = {k: v for k, v in lag_corrs.items() if k != 0}
    if non_sync_lags:
        optimal_lag = max(non_sync_lags.items(), key=lambda x: abs(x[1]))[0]
    else:
        optimal_lag = 0
    
    # Compute aggregated metric
    I_i_leads_j = sum(abs(lag_corrs.get(lag, 0)) for lag in range(1, max_lag + 1))
    I_j_leads_i = sum(abs(lag_corrs.get(-lag, 0)) for lag in range(1, max_lag + 1))
    
    if I_i_leads_j + I_j_leads_i == 0:
        return 0.0, 0, lag_corrs, is_sync
    
    lead_lag_metric = (
        np.sign(I_i_leads_j - I_j_leads_i) * 
        max(I_i_leads_j, I_j_leads_i) / 
        (I_i_leads_j + I_j_leads_i)
    )
    
    return lead_lag_metric, optimal_lag, lag_corrs, is_sync


def compute_lead_lag_matrix_with_control(returns_df: pd.DataFrame, tickers: List[str],
                                         max_lag: int = 5, n_jobs: int = -1) -> Tuple:
    """
    Compute lead-lag matrix with partial correlation control.
    
    KEY ENHANCEMENT: Uses partial correlation to remove spurious lead-lag.
    
    Returns:
    - lead_lag_df: Lead-lag metrics (controlled)
    - optimal_lags_df: Optimal lag for each pair
    - all_lag_correlations: All correlations at each lag
    - filtered_pairs: Pairs with true lead-lag (not synchronous)
    """
    print(f"\n{'='*60}")
    print("SECTION 2: LEAD-LAG COMPUTATION WITH PARTIAL CORRELATION")
    print(f"{'='*60}\n")
    print("Method: ccf-auc with distance correlation")
    print("Enhancement: Partial correlation control for spurious lead-lag")
    print(f"Maximum lag: {max_lag} days")
    
    returns_dict = {}
    for ticker in tickers:
        returns_dict[ticker] = returns_df[returns_df['ticker'] == ticker]['returns'].values
    
    def compute_pair(i, j):
        returns_i = returns_dict[tickers[i]]
        returns_j = returns_dict[tickers[j]]
        return compute_lead_lag_with_partial_correlation(returns_i, returns_j, max_lag)
    
    print("\nComputing pairwise lead-lag relationships...")
    n = len(tickers)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_pair)(i, j) for i, j in tqdm(pairs, desc="Computing")
    )
    
    # Unpack results
    lead_lag_matrix = np.zeros((n, n))
    optimal_lags_matrix = np.zeros((n, n), dtype=int)
    all_lag_correlations = {}
    filtered_pairs = []
    synchronous_count = 0
    
    for (i, j), (metric, opt_lag, lag_corrs, is_sync) in zip(pairs, results):
        if not is_sync:
            lead_lag_matrix[i, j] = metric
            lead_lag_matrix[j, i] = -metric
            optimal_lags_matrix[i, j] = opt_lag
            optimal_lags_matrix[j, i] = -opt_lag
            all_lag_correlations[(tickers[i], tickers[j])] = lag_corrs
            filtered_pairs.append((i, j))
        else:
            synchronous_count += 1
    
    lead_lag_df = pd.DataFrame(lead_lag_matrix, index=tickers, columns=tickers)
    optimal_lags_df = pd.DataFrame(optimal_lags_matrix, index=tickers, columns=tickers)
    
    print(f"\nFiltering Results:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Synchronous (filtered): {synchronous_count}")
    print(f"  True lead-lag pairs: {len(filtered_pairs)}")
    print(f"  Percentage with lead-lag: {len(filtered_pairs)/len(pairs)*100:.1f}%")
    
    return lead_lag_df, optimal_lags_df, all_lag_correlations, filtered_pairs


# ============================================================================
# SECTION 3: STATISTICAL SIGNIFICANCE TESTING (FDR Correction)
# ============================================================================

def bootstrap_pvalue(returns_i: np.ndarray, returns_j: np.ndarray,
                     observed_metric: float, max_lag: int = 5,
                     n_bootstrap: int = N_BOOTSTRAP) -> float:
    """
    Compute p-value using bootstrap permutation test.
    
    Null hypothesis: No lead-lag relationship
    Test: Shuffle returns_j to break temporal structure
    """
    null_distribution = []
    
    for _ in range(n_bootstrap):
        shuffled_j = np.random.permutation(returns_j)
        null_metric, _, _, _ = compute_lead_lag_with_partial_correlation(
            returns_i, shuffled_j, max_lag
        )
        null_distribution.append(abs(null_metric))
    
    # Two-sided p-value
    p_value = np.mean(np.array(null_distribution) >= abs(observed_metric))
    
    return p_value


def apply_fdr_correction(lead_lag_df: pd.DataFrame, returns_df: pd.DataFrame,
                        tickers: List[str], alpha: float = FDR_ALPHA,
                        max_lag: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply False Discovery Rate correction using bootstrap p-values.
    
    This controls for multiple testing across all stock pairs.
    
    Returns:
    - fdr_corrected_matrix: Lead-lag matrix with only FDR-significant pairs
    - p_value_matrix: P-values for each pair
    """
    print(f"\n{'='*60}")
    print("SECTION 3: STATISTICAL SIGNIFICANCE (FDR Correction)")
    print(f"{'='*60}\n")
    print(f"FDR level: {alpha}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    
    returns_dict = {}
    for ticker in tickers:
        returns_dict[ticker] = returns_df[returns_df['ticker'] == ticker]['returns'].values
    
    # Collect non-zero pairs
    n = len(tickers)
    pairs_to_test = []
    observed_metrics = []
    
    for i in range(n):
        for j in range(i+1, n):
            if lead_lag_df.iloc[i, j] != 0:
                pairs_to_test.append((i, j))
                observed_metrics.append(lead_lag_df.iloc[i, j])
    
    if not pairs_to_test:
        print("No pairs to test (all filtered as synchronous)")
        return lead_lag_df, pd.DataFrame(index=tickers, columns=tickers)
    
    print(f"\nComputing p-values for {len(pairs_to_test)} pairs...")
    
    def compute_pval(idx):
        i, j = pairs_to_test[idx]
        returns_i = returns_dict[tickers[i]]
        returns_j = returns_dict[tickers[j]]
        observed = observed_metrics[idx]
        return bootstrap_pvalue(returns_i, returns_j, observed, max_lag)
    
    p_values = Parallel(n_jobs=-1)(
        delayed(compute_pval)(idx) for idx in tqdm(range(len(pairs_to_test)))
    )
    
    # Apply Benjamini-Hochberg FDR
    print("\nApplying Benjamini-Hochberg FDR correction...")
    reject, adjusted_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # Create corrected matrices
    fdr_matrix = pd.DataFrame(0.0, index=tickers, columns=tickers)
    p_value_matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    
    for idx, (i, j) in enumerate(pairs_to_test):
        p_value_matrix.iloc[i, j] = p_values[idx]
        p_value_matrix.iloc[j, i] = p_values[idx]
        
        if reject[idx]:
            fdr_matrix.iloc[i, j] = lead_lag_df.iloc[i, j]
            fdr_matrix.iloc[j, i] = lead_lag_df.iloc[j, i]
    
    n_significant = reject.sum()
    print(f"\nPairs surviving FDR correction: {n_significant}/{len(pairs_to_test)} "
          f"({n_significant/len(pairs_to_test)*100:.1f}%)")
    
    return fdr_matrix, p_value_matrix


# ============================================================================
# SECTION 4: TEMPORAL PERSISTENCE VALIDATION
# ============================================================================

def split_into_windows(returns_df: pd.DataFrame, 
                       window_size: str = '1Y', 
                       step: str = '6M') -> List[Dict]:
    """
    Split data into overlapping time windows.
    
    This allows us to check if lead-lag relationships persist over time.
    """
    windows = []
    dates = returns_df.index.unique().sort_values()
    start_date = dates[0]
    end_date = dates[-1]
    
    current_start = start_date
    window_offset = pd.DateOffset(years=1)
    step_offset = pd.DateOffset(months=6)

    while current_start + window_offset <= end_date:
        current_end = current_start + window_offset
        window_data = returns_df[
            (returns_df.index >= current_start) & 
            (returns_df.index < current_end)
        ]
        windows.append({
            'data': window_data,
            'start': current_start,
            'end': current_end
        })
        current_start += step_offset
    
    return windows


def compute_temporal_persistence(returns_df: pd.DataFrame, tickers: List[str],
                                 window_size: str = '1Y', step: str = '6M',
                                 max_lag: int = 5) -> pd.DataFrame:
    """
    Validate temporal persistence of lead-lag relationships.
    
    For each window:
    - Recompute lead-lag relationships
    - Track which pairs remain significant
    
    Returns:
    - persistence_df: For each pair, % of windows where relationship exists
    """
    print(f"\n{'='*60}")
    print("SECTION 4: TEMPORAL PERSISTENCE VALIDATION")
    print(f"{'='*60}\n")
    print(f"Window size: {window_size}")
    print(f"Step size: {step}")
    print(f"Minimum persistence threshold: {MIN_PERSISTENCE*100:.0f}%")
    
    windows = split_into_windows(returns_df, window_size, step)
    print(f"\nAnalyzing {len(windows)} time windows...")
    
    # Compute lead-lag for each window
    window_matrices = []
    for window in tqdm(windows, desc="Processing windows"):
        if len(window['data']) < 100:  # Skip windows with insufficient data
            continue
        
        # Compute lead-lag (without FDR to save time)
        lead_lag_df, _, _, _ = compute_lead_lag_matrix_with_control(
            window['data'], tickers, max_lag=max_lag, n_jobs=-1
        )
        window_matrices.append(lead_lag_df)
    
    print(f"Analyzed {len(window_matrices)} valid windows")
    
    # Compute persistence
    n = len(tickers)
    persistence_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Count windows where this pair is non-zero
            count = sum(1 for wm in window_matrices if wm.iloc[i, j] != 0)
            persistence = count / len(window_matrices) if window_matrices else 0
            persistence_matrix[i, j] = persistence
            persistence_matrix[j, i] = persistence
    
    persistence_df = pd.DataFrame(persistence_matrix, index=tickers, columns=tickers)
    
    # Statistics
    stable_pairs = (persistence_df >= MIN_PERSISTENCE)
    n_stable = stable_pairs.sum().sum() // 2
    total_pairs = n * (n - 1) // 2
    
    print(f"\nPersistence Results:")
    print(f"  Pairs with ≥{MIN_PERSISTENCE*100:.0f}% persistence: {n_stable}/{total_pairs}")
    
    return persistence_df


# ============================================================================
# SECTION 5: TOP PAIRS & ANALYSIS
# ============================================================================

def identify_top_pairs(lead_lag_df: pd.DataFrame, optimal_lags_df: pd.DataFrame,
                      persistence_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Identify top N strongest and most persistent lead-lag pairs.
    """
    print(f"\n{'='*60}")
    print(f"SECTION 5: TOP {top_n} LEAD-LAG PAIRS")
    print(f"{'='*60}\n")
    
    pairs_data = []
    n = len(lead_lag_df)
    
    for i in range(n):
        for j in range(i+1, n):
            if lead_lag_df.iloc[i, j] != 0:
                strength = lead_lag_df.iloc[i, j]
                opt_lag = optimal_lags_df.iloc[i, j]
                persistence = persistence_df.iloc[i, j]
                
                if strength > 0:
                    pairs_data.append({
                        'leader': lead_lag_df.index[i],
                        'lagger': lead_lag_df.columns[j],
                        'strength': strength,
                        'optimal_lag': opt_lag,
                        'persistence': persistence,
                        'direction': f'{lead_lag_df.index[i]} → {lead_lag_df.columns[j]}'
                    })
                else:
                    pairs_data.append({
                        'leader': lead_lag_df.columns[j],
                        'lagger': lead_lag_df.index[i],
                        'strength': abs(strength),
                        'optimal_lag': abs(opt_lag),
                        'persistence': persistence,
                        'direction': f'{lead_lag_df.columns[j]} → {lead_lag_df.index[i]}'
                    })
    
    pairs_df = pd.DataFrame(pairs_data)
    
    # Sort by combination of strength and persistence
    pairs_df['score'] = pairs_df['strength'] * pairs_df['persistence']
    pairs_df = pairs_df.sort_values('score', ascending=False).head(top_n)
    pairs_df = pairs_df.drop('score', axis=1)
    pairs_df = pairs_df.reset_index(drop=True)
    pairs_df.index = pairs_df.index + 1
    
    print("Top lead-lag pairs (by strength × persistence):")
    print(pairs_df.to_string())
    
    return pairs_df


# ============================================================================
# SECTION 6: VISUALIZATIONS (Keep All - These Are Excellent)
# ============================================================================

def find_comovement_windows(returns_i: np.ndarray, returns_j: np.ndarray,
                           lag: int, window_size: int = 20,
                           correlation_threshold: float = 0.5) -> List[Tuple[int, int]]:
    """
    Find time windows where returns show strong co-movement at specified lag.
    
    This is essentially temporal validation at a granular level.
    """
    n = len(returns_i)
    comovement_windows = []
    
    for start in range(0, n - window_size - abs(lag)):
        if lag >= 0:
            window_i = returns_i[start:start + window_size]
            window_j = returns_j[start + lag:start + lag + window_size]
        else:
            window_i = returns_i[start - lag:start - lag + window_size]
            window_j = returns_j[start:start + window_size]
        
        try:
            corr = distance_correlation(window_i, window_j)
            if corr > correlation_threshold:
                comovement_windows.append((start, start + window_size))
        except:
            pass
    
    # Merge overlapping windows
    if not comovement_windows:
        return []
    
    merged = [comovement_windows[0]]
    for start, end in comovement_windows[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    
    return merged


def create_case_study_visualization(returns_df: pd.DataFrame, 
                                    leader: str, lagger: str,
                                    optimal_lag: int,
                                    all_lag_correlations: Dict,
                                    persistence: float,
                                    save_dir: str = './results/'):
    """
    Create detailed case study visualization for a stock pair.
    
    Shows:
    1. Price charts with highlighted co-movement periods
    2. Returns correlation at different lags
    3. Scatter plot showing the relationship
    4. Statistics
    """
    print(f"\n  Creating case study: {leader} → {lagger} (lag={optimal_lag})")
    
    leader_data = returns_df[returns_df['ticker'] == leader].copy()
    lagger_data = returns_df[returns_df['ticker'] == lagger].copy()
    
    common_dates = leader_data.index.intersection(lagger_data.index)
    leader_data = leader_data.loc[common_dates]
    lagger_data = lagger_data.loc[common_dates]
    
    lag_corrs = all_lag_correlations.get((leader, lagger)) or \
                all_lag_correlations.get((lagger, leader), {})
    
    comovement_windows = find_comovement_windows(
        leader_data['returns'].values,
        lagger_data['returns'].values,
        optimal_lag
    )
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Leader price chart
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(leader_data.index, leader_data['close'], 
             color='lightgray', linewidth=1, alpha=0.5, label='Full period')
    
    for start_idx, end_idx in comovement_windows:
        dates_window = common_dates[start_idx:end_idx]
        prices_window = leader_data.loc[dates_window, 'close']
        ax1.plot(dates_window, prices_window, color='blue', linewidth=2)
    
    ax1.set_title(f'{leader} Price (Leader) - Blue = Strong Co-movement Periods',
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # 2. Lagger price chart (shifted)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(lagger_data.index, lagger_data['close'],
             color='lightgray', linewidth=1, alpha=0.5, label='Full period')
    
    for start_idx, end_idx in comovement_windows:
        dates_window = common_dates[start_idx:end_idx]
        shifted_dates = dates_window + pd.Timedelta(days=optimal_lag)
        
        mask = lagger_data.index.isin(shifted_dates)
        if mask.any():
            ax2.plot(lagger_data.index[mask], lagger_data.loc[mask, 'close'],
                    color='red', linewidth=2)
    
    ax2.set_title(f'{lagger} Price (Lagger, +{optimal_lag} days) - Red = Predicted Movement',
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price ($)', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # 3. Lag correlation profile
    ax3 = fig.add_subplot(gs[2, 0])
    if lag_corrs:
        lags = sorted(lag_corrs.keys())
        corrs = [lag_corrs[lag] for lag in lags]
        
        colors = ['red' if lag == optimal_lag else 'gray' for lag in lags]
        ax3.bar(lags, corrs, color=colors, edgecolor='black', linewidth=1.5)
        ax3.axvline(0, color='black', linestyle='--', alpha=0.5, label='Synchronous')
        ax3.axvline(optimal_lag, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal = {optimal_lag}')
        
        ax3.set_xlabel('Lag (days)', fontsize=12)
        ax3.set_ylabel('Partial Distance Correlation', fontsize=12)
        ax3.set_title('Correlation at Different Lags (Controlled)', 
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
    
    # 4. Scatter plot
    ax4 = fig.add_subplot(gs[2, 1])
    if optimal_lag >= 0:
        x_data = leader_data['returns'].values[:-optimal_lag if optimal_lag > 0 else None]
        y_data = lagger_data['returns'].values[optimal_lag:]
        x_label = f'{leader} returns (t)'
        y_label = f'{lagger} returns (t+{optimal_lag})'
    else:
        x_data = lagger_data['returns'].values[:optimal_lag]
        y_data = leader_data['returns'].values[-optimal_lag:]
        x_label = f'{lagger} returns (t)'
        y_label = f'{leader} returns (t+{abs(optimal_lag)})'
    
    ax4.scatter(x_data, y_data, alpha=0.3, s=10)
    ax4.set_xlabel(x_label, fontsize=12)
    ax4.set_ylabel(y_label, fontsize=12)
    ax4.set_title('Returns Relationship at Optimal Lag', 
                 fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax4.plot(sorted(x_data), p(sorted(x_data)), "r-", linewidth=2, alpha=0.8)
    
    # 5. Statistics
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    stats_text = f"""
    RELATIONSHIP STATISTICS
    
    Leader: {leader}
    Lagger: {lagger}
    Optimal Lag: {optimal_lag} days
    
    Partial Distance Correlation (optimal lag): {lag_corrs.get(optimal_lag, 0):.4f}
    Synchronous Correlation (lag=0): {lag_corrs.get(0, 0):.4f}
    Temporal Persistence: {persistence:.1%}
    
    Co-movement Periods: {len(comovement_windows)} windows found
    Total Days in Co-movement: {sum(end - start for start, end in comovement_windows)}
    
    Interpretation:
    • {leader} movements predict {lagger} movements {optimal_lag} days later
    • This relationship persisted in {persistence:.0%} of time windows analyzed
    • Correlation strength: {'Strong' if lag_corrs.get(optimal_lag, 0) > 0.5 else 'Moderate'}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(f'{save_dir}case_study_{leader}_to_{lagger}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_lag_distribution_plot(optimal_lags_df: pd.DataFrame, 
                                 save_dir: str = './results/'):
    """Visualize distribution of optimal lags."""
    print("\n  Creating lag distribution plot...")
    
    lags = optimal_lags_df.values[np.triu_indices_from(optimal_lags_df.values, k=1)]
    lags = lags[lags != 0]
    
    if len(lags) == 0:
        print("    No non-zero lags to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(lags, bins=range(min(lags)-1, max(lags)+2), 
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, 
                   label='Synchronous (excluded)')
    axes[0].set_xlabel('Optimal Lag (days)', fontsize=12)
    axes[0].set_ylabel('Number of Pairs', fontsize=12)
    axes[0].set_title('Distribution of Optimal Lags', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    
    # Bar chart
    lag_counts = pd.Series(lags).value_counts().sort_index()
    colors = ['red' if lag > 0 else 'blue' for lag in lag_counts.index]
    axes[1].bar(lag_counts.index, lag_counts.values, color=colors, edgecolor='black')
    axes[1].set_xlabel('Lag (days)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Lead-Lag Direction (Red=Positive, Blue=Negative)',
                     fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}lag_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    STREAMLINED PIPELINE:
    1. Enhanced data preprocessing (quality checks)
    2. Partial correlation lead-lag computation (removes spurious)
    3. FDR correction (controls multiple testing)
    4. Temporal persistence (validates stability)
    5. Visualizations (case studies, distributions)
    """
    print("\n" + "="*70)
    print(" STREAMLINED LEAD-LAG DETECTION")
    print(" Focus: Accurate Pairwise Relationships")
    print(" Enhancements: Partial Correlation + FDR + Temporal Validation")
    print("="*70)
    
    import os
    os.makedirs('./results/', exist_ok=True)
    
    # Step 1: Load and preprocess data
    # tickers = get_sp500_tickers(n_stocks=50)
    tickers = get_russell3000_tickers(n_stocks = 2000)
    returns_df, successful_tickers = load_and_preprocess_data(
        tickers, START_DATE, END_DATE
    )
    
    # Step 2: Compute lead-lag with partial correlation control
    lead_lag_df, optimal_lags_df, all_lag_correlations, filtered_pairs = \
        compute_lead_lag_matrix_with_control(
            returns_df, successful_tickers, max_lag=MAX_LAG
        )
    
    # Step 3: Apply FDR correction
    user_input = input("\nApply FDR correction? (recommended, takes ~10 min) [y/n]: ")
    if user_input.lower() == 'y':
        fdr_matrix, p_value_matrix = apply_fdr_correction(
            lead_lag_df, returns_df, successful_tickers, alpha=FDR_ALPHA, max_lag=MAX_LAG
        )
    else:
        print("Skipping FDR correction")
        fdr_matrix = lead_lag_df
        p_value_matrix = pd.DataFrame(index=successful_tickers, columns=successful_tickers)
    
    # Step 4: Temporal persistence validation
    persistence_df = compute_temporal_persistence(
        returns_df, successful_tickers, window_size='1Y', step='6M', max_lag=MAX_LAG
    )
    
    # Filter by persistence
    final_matrix = fdr_matrix.copy()
    final_matrix.values[persistence_df.values < MIN_PERSISTENCE] = 0
    
    print(f"\n{'='*70}")
    print("FINAL FILTERING SUMMARY")
    print(f"{'='*70}\n")
    total_pairs = len(successful_tickers) * (len(successful_tickers) - 1) // 2
    print(f"Total pairs: {total_pairs}")
    print(f"After sync filter: {(lead_lag_df != 0).sum().sum() // 2}")
    print(f"After FDR correction: {(fdr_matrix != 0).sum().sum() // 2}")
    print(f"After persistence filter: {(final_matrix != 0).sum().sum() // 2}")
    
    # Step 5: Identify top pairs
    top_pairs = identify_top_pairs(
        final_matrix, optimal_lags_df, persistence_df, top_n=10
    )
    
    # Step 6: Create visualizations
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    create_lag_distribution_plot(optimal_lags_df)
    
    print("\nCase studies for top 3 pairs:")
    for idx in range(min(3, len(top_pairs))):
        create_case_study_visualization(
            returns_df,
            top_pairs.iloc[idx]['leader'],
            top_pairs.iloc[idx]['lagger'],
            top_pairs.iloc[idx]['optimal_lag'],
            all_lag_correlations,
            top_pairs.iloc[idx]['persistence']
        )
    
    # Step 7: Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    final_matrix.to_csv('./results/lead_lag_matrix_final.csv')
    optimal_lags_df.to_csv('./results/optimal_lags.csv')
    persistence_df.to_csv('./results/persistence_scores.csv')
    top_pairs.to_csv('./results/top_pairs.csv')
    p_value_matrix.to_csv('./results/p_values.csv')
    
    with open('./results/summary.txt', 'w', encoding='utf-8') as f:
        f.write("LEAD-LAG ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Stocks: {len(successful_tickers)}\n")
        f.write(f"Period: {START_DATE} to {END_DATE}\n\n")
        
        f.write("FILTERS APPLIED:\n")
        f.write(f"  1. Data quality (volume, missing data, outliers)\n")
        f.write(f"  2. Partial correlation (removes spurious lead-lag)\n")
        f.write(f"  3. Synchronous filtering\n")
        f.write(f"  4. FDR correction (alpha={FDR_ALPHA})\n")
        f.write(f"  5. Temporal persistence (less than or equal to{MIN_PERSISTENCE*100:.0f}%)\n\n")
        
        f.write("TOP 10 PAIRS:\n")
        f.write(top_pairs.to_string())
    
    print("✓ Complete! Files saved:")
    print("  - lead_lag_matrix_final.csv")
    print("  - optimal_lags.csv")
    print("  - persistence_scores.csv")
    print("  - top_pairs.csv")
    print("  - p_values.csv")
    print("  - lag_distribution.png")
    print("  - case_study_*.png (top 3 pairs)")
    print("  - summary.txt")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()