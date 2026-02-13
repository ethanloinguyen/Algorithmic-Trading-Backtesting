"""
LEAD-LAG DETECTION FRAMEWORK
Dynamic Rolling Window Version with Advanced Analytics

This script implements a comprehensive lead-lag relationship detection system
for stock pairs using distance correlation, permutation testing, and stability analysis.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.api import OLS, add_constant
import dcor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import pickle
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import json

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
CONFIG = {
    'n_stocks': 500,  # Top 500 Russell 3000 stocks
    'start_date': '2000-01-01',
    'end_date': '2026-01-01',
    'window_years': 3,
    'step_years': 1,
    'lags': [1, 2, 3, 4, 5],  # Exclude 0-lag
    'n_permutations': 500,
    'block_size': 60,  # For block permutation
    'fdr_alpha': 0.05,
    'top_percentile': 0.05,  # Top 5%
    'stability_threshold': 0.6,
    'train_end': '2015-12-31',
    'test_start': '2016-01-01',
}

class LeadLagFramework:
    """Main framework for lead-lag detection and analysis"""
    
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.results = {}
        self.output_dir = Path('./lead_lag_results')
        self.output_dir.mkdir(exist_ok=True)
        
    def download_data(self):
        """Download stock data for top Russell 3000 stocks"""
        print("=" * 80)
        print("STEP 1: DATA DOWNLOAD AND PREPARATION")
        print("=" * 80)
        
        # Get Russell 3000 components (using S&P 500 as proxy + additional large caps)
        # In production, this would query your BigQuery database
        print("\nFetching Russell 3000 component list...")
        
        # Download S&P 500 tickers as base
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        try:
            sp500_table = pd.read_html(sp500_url)[0]
            tickers = sp500_table['Symbol'].str.replace('.', '-').tolist()
        except:
            print("Warning: Could not fetch S&P 500 list, using predefined list")
            tickers = self._get_fallback_tickers()
        
        # Take top N stocks
        tickers = tickers[:self.config['n_stocks']]
        
        print(f"\nDownloading data for {len(tickers)} stocks...")
        print(f"Date range: {self.config['start_date']} to {self.config['end_date']}")
        
        # Download Russell 3000 index data (using ^RUA as proxy)
        print("\nDownloading Russell 3000 index...")
        try:
            market_data = yf.download('^RUA', 
                                     start=self.config['start_date'],
                                     end=self.config['end_date'],
                                     progress=False)
            self.data['market'] = market_data['Adj Close']
            print(f"✓ Market index downloaded: {len(self.data['market'])} days")
        except Exception as e:
            print(f"Error downloading market index: {e}")
            print("Using synthetic market index from stock data...")
        
        # Download stock data
        stock_data = {}
        failed_tickers = []
        
        for ticker in tqdm(tickers, desc="Downloading stocks"):
            try:
                df = yf.download(ticker,
                               start=self.config['start_date'],
                               end=self.config['end_date'],
                               progress=False)
                if len(df) > 0:
                    stock_data[ticker] = df['Adj Close']
            except Exception as e:
                failed_tickers.append(ticker)
        
        print(f"\n✓ Successfully downloaded {len(stock_data)} stocks")
        if failed_tickers:
            print(f"✗ Failed: {len(failed_tickers)} stocks")
        
        # Convert to DataFrame
        self.data['prices'] = pd.DataFrame(stock_data)
        
        # If market index download failed, create from stock data
        if 'market' not in self.data:
            self.data['market'] = self.data['prices'].mean(axis=1)
            print("Created synthetic market index from stock data")
        
        return self.data['prices']
    
    def _get_fallback_tickers(self):
        """Fallback ticker list if download fails"""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
                'ABBV', 'KO', 'PEP', 'AVGO', 'COST', 'TMO', 'WMT', 'MCD', 'CSCO',
                'ACN', 'ABT', 'DHR', 'ADBE', 'NFLX', 'LIN', 'NKE', 'DIS', 'VZ',
                'CRM', 'CMCSA', 'TXN', 'NEE', 'PM', 'UPS', 'RTX', 'INTC', 'ORCL',
                'QCOM', 'HON', 'IBM', 'INTU', 'UNP', 'LOW', 'AMD', 'AMGN', 'CAT']
    
    def filter_and_prepare(self):
        """Step 1: Initial data preparation"""
        print("\n" + "=" * 80)
        print("STEP 1.1: LIQUIDITY & QUALITY FILTERING")
        print("=" * 80)
        
        prices = self.data['prices']
        
        # Count missing days per stock
        missing_days = prices.isna().sum()
        print(f"\nMissing days per stock - Max: {missing_days.max()}, Mean: {missing_days.mean():.1f}")
        
        # Filter: exclude stocks with >100 missing days
        valid_stocks = missing_days[missing_days <= 100].index.tolist()
        print(f"Stocks after missing data filter: {len(valid_stocks)}/{len(prices.columns)}")
        
        # Forward fill small gaps
        prices = prices[valid_stocks].fillna(method='ffill', limit=5)
        
        # Compute returns
        print("\n" + "=" * 80)
        print("STEP 1.2: COMPUTING LOG RETURNS")
        print("=" * 80)
        
        returns = np.log(prices / prices.shift(1)).dropna()
        market_returns = np.log(self.data['market'] / self.data['market'].shift(1)).dropna()
        
        # Align dates
        common_dates = returns.index.intersection(market_returns.index)
        returns = returns.loc[common_dates]
        market_returns = market_returns.loc[common_dates]
        
        # Filter by volatility (exclude extremely low volatility)
        volatility = returns.std()
        valid_stocks = volatility[volatility > volatility.quantile(0.05)].index.tolist()
        returns = returns[valid_stocks]
        
        print(f"\n✓ Final stock universe: {len(returns.columns)} stocks")
        print(f"✓ Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"✓ Total trading days: {len(returns)}")
        
        self.data['returns'] = returns
        self.data['market_returns'] = market_returns
        
        return returns
    
    def create_rolling_windows(self):
        """Step 2: Define rolling windows"""
        print("\n" + "=" * 80)
        print("STEP 2: DEFINING ROLLING WINDOWS")
        print("=" * 80)
        
        returns = self.data['returns']
        window_days = self.config['window_years'] * 252  # Trading days
        step_days = self.config['step_years'] * 252
        
        windows = []
        start_idx = 0
        
        while start_idx + window_days <= len(returns):
            end_idx = start_idx + window_days
            window_data = {
                'start_date': returns.index[start_idx],
                'end_date': returns.index[end_idx - 1],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'returns': returns.iloc[start_idx:end_idx],
                'market_returns': self.data['market_returns'].iloc[start_idx:end_idx]
            }
            windows.append(window_data)
            start_idx += step_days
        
        print(f"\n✓ Created {len(windows)} rolling windows")
        print(f"  Window length: {self.config['window_years']} years ({window_days} days)")
        print(f"  Step size: {self.config['step_years']} year(s) ({step_days} days)")
        
        for i, w in enumerate(windows[:3]):
            print(f"  Window {i+1}: {w['start_date'].date()} to {w['end_date'].date()}")
        if len(windows) > 3:
            print(f"  ... ({len(windows) - 3} more windows)")
        
        self.data['windows'] = windows
        return windows
    
    def process_window(self, window_data, window_idx):
        """Step 3: Process a single rolling window"""
        returns = window_data['returns']
        market_returns = window_data['market_returns']
        
        # 3.1 Factor Residualization
        residuals = {}
        betas = {}
        
        for stock in returns.columns:
            y = returns[stock].values
            X = add_constant(market_returns.values)
            
            # Handle NaN values
            valid_idx = ~(np.isnan(y) | np.isnan(market_returns.values))
            if valid_idx.sum() < 100:  # Minimum observations
                continue
            
            try:
                model = OLS(y[valid_idx], X[valid_idx]).fit()
                alpha_i = model.params[0]
                beta_i = model.params[1]
                
                # Compute residuals
                epsilon = y - (alpha_i + beta_i * market_returns.values)
                residuals[stock] = epsilon
                betas[stock] = beta_i
            except:
                continue
        
        # 3.2 Standardization
        z_scores = {}
        for stock, resid in residuals.items():
            valid_resid = resid[~np.isnan(resid)]
            if len(valid_resid) < 100:
                continue
            mu = valid_resid.mean()
            sigma = valid_resid.std()
            if sigma > 0:
                z = (resid - mu) / sigma
                z_scores[stock] = z
        
        return z_scores, betas
    
    def compute_distance_correlation(self, x, y, lag):
        """Compute distance correlation at a specific lag"""
        if lag >= len(x):
            return 0.0
        
        x_lagged = x[:-lag] if lag > 0 else x
        y_shifted = y[lag:] if lag > 0 else y
        
        # Remove NaN
        valid_idx = ~(np.isnan(x_lagged) | np.isnan(y_shifted))
        if valid_idx.sum() < 50:  # Minimum observations
            return 0.0
        
        x_clean = x_lagged[valid_idx]
        y_clean = y_shifted[valid_idx]
        
        try:
            dcor_value = dcor.distance_correlation(x_clean, y_clean)
            return dcor_value
        except:
            return 0.0
    
    def compute_pearson_at_lag(self, x, y, lag):
        """Compute Pearson correlation at specific lag"""
        if lag >= len(x):
            return 0.0, 1.0
        
        x_lagged = x[:-lag] if lag > 0 else x
        y_shifted = y[lag:] if lag > 0 else y
        
        valid_idx = ~(np.isnan(x_lagged) | np.isnan(y_shifted))
        if valid_idx.sum() < 50:
            return 0.0, 1.0
        
        try:
            corr, pval = pearsonr(x_lagged[valid_idx], y_shifted[valid_idx])
            return corr, pval
        except:
            return 0.0, 1.0
    
    def block_permutation(self, series, block_size):
        """Perform block permutation preserving autocorrelation"""
        n = len(series)
        n_blocks = n // block_size
        
        # Create blocks
        blocks = [series[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        remainder = series[n_blocks*block_size:]
        
        # Shuffle blocks
        np.random.shuffle(blocks)
        
        # Reconstruct series
        permuted = np.concatenate(blocks + [remainder])
        return permuted
    
    def detect_lead_lag_pairs(self):
        """Step 4: Lead-lag detection across all windows"""
        print("\n" + "=" * 80)
        print("STEP 4: LEAD-LAG DETECTION")
        print("=" * 80)
        
        windows = self.data['windows']
        all_results = []
        
        for window_idx, window_data in enumerate(tqdm(windows, desc="Processing windows")):
            print(f"\nWindow {window_idx + 1}/{len(windows)}: "
                  f"{window_data['start_date'].date()} to {window_data['end_date'].date()}")
            
            # Process window
            z_scores, betas = self.process_window(window_data, window_idx)
            stocks = list(z_scores.keys())
            
            if len(stocks) < 2:
                print(f"  Warning: Only {len(stocks)} valid stocks in window")
                continue
            
            print(f"  Valid stocks: {len(stocks)}")
            print(f"  Total pairs to evaluate: {len(stocks) * (len(stocks) - 1)}")
            
            # Evaluate all pairs
            pair_results = []
            stock_pairs = [(a, b) for a in stocks for b in stocks if a != b]
            
            for stock_a, stock_b in tqdm(stock_pairs, desc=f"  Evaluating pairs", leave=False):
                z_a = z_scores[stock_a]
                z_b = z_scores[stock_b]
                
                # Compute distance correlation for each lag
                dcor_values = []
                for lag in self.config['lags']:
                    dcor_val = self.compute_distance_correlation(z_a, z_b, lag)
                    dcor_values.append(dcor_val)
                
                # Compute AUC
                auc = np.sum(np.abs(dcor_values))
                
                if auc == 0:
                    continue
                
                # Find optimal lag
                optimal_lag_idx = np.argmax(np.abs(dcor_values))
                optimal_lag = self.config['lags'][optimal_lag_idx]
                
                # Compute Pearson correlation at optimal lag
                pearson_corr, pearson_pval = self.compute_pearson_at_lag(z_a, z_b, optimal_lag)
                
                # Permutation testing
                perm_aucs = []
                for _ in range(self.config['n_permutations']):
                    z_b_perm = self.block_permutation(z_b, self.config['block_size'])
                    
                    perm_dcors = []
                    for lag in self.config['lags']:
                        dcor_val = self.compute_distance_correlation(z_a, z_b_perm, lag)
                        perm_dcors.append(dcor_val)
                    
                    perm_auc = np.sum(np.abs(perm_dcors))
                    perm_aucs.append(perm_auc)
                
                # Compute empirical p-value
                p_value = np.sum(np.array(perm_aucs) >= auc) / self.config['n_permutations']
                
                pair_results.append({
                    'window_idx': window_idx,
                    'leader': stock_a,
                    'lagger': stock_b,
                    'auc': auc,
                    'optimal_lag': optimal_lag,
                    'pearson_corr': pearson_corr,
                    'pearson_pval': pearson_pval,
                    'perm_pval': p_value,
                    'dcor_values': dcor_values,
                    'start_date': window_data['start_date'],
                    'end_date': window_data['end_date']
                })
            
            # FDR correction within window
            if pair_results:
                p_values = [r['perm_pval'] for r in pair_results]
                from statsmodels.stats.multitest import fdrcorrection
                reject, pvals_corrected = fdrcorrection(p_values, alpha=self.config['fdr_alpha'])
                
                for i, result in enumerate(pair_results):
                    result['fdr_pval'] = pvals_corrected[i]
                    result['significant'] = reject[i]
                
                # Keep only significant pairs
                significant = [r for r in pair_results if r['significant']]
                
                # Apply strength threshold (top percentile)
                if significant:
                    auc_threshold = np.percentile([r['auc'] for r in significant],
                                                 100 * (1 - self.config['top_percentile']))
                    significant = [r for r in significant if r['auc'] >= auc_threshold]
                
                print(f"  Significant pairs: {len(significant)}")
                all_results.extend(significant)
        
        self.results['pair_results'] = pd.DataFrame(all_results)
        print(f"\n✓ Total significant pairs across all windows: {len(all_results)}")
        
        return self.results['pair_results']
    
    def compute_stability_metrics(self):
        """Step 5: Stability filtering across windows"""
        print("\n" + "=" * 80)
        print("STEP 5: STABILITY FILTERING")
        print("=" * 80)
        
        df = self.results['pair_results']
        
        # Group by pair
        stability_results = []
        
        for (leader, lagger), group in tqdm(df.groupby(['leader', 'lagger']),
                                           desc="Computing stability metrics"):
            n_windows = len(self.data['windows'])
            n_significant = len(group)
            
            # S1: Significance Persistence
            s1 = n_significant / n_windows
            
            # S2: Lag Stability (lower variance = higher stability)
            lag_variance = group['optimal_lag'].var() if len(group) > 1 else 0
            s2 = 1 / (1 + lag_variance)  # Normalize
            
            # S3: Sign Consistency
            signs = np.sign(group['pearson_corr'])
            most_common_sign = stats.mode(signs, keepdims=True)[0][0]
            s3 = (signs == most_common_sign).sum() / len(signs)
            
            # S4: Strength Stability (AUC coefficient of variation)
            auc_cv = group['auc'].std() / group['auc'].mean() if group['auc'].mean() > 0 else float('inf')
            s4 = 1 / (1 + auc_cv)
            
            # Final stability score (weighted combination)
            stability_score = 0.4 * s1 + 0.2 * s2 + 0.2 * s3 + 0.2 * s4
            
            stability_results.append({
                'leader': leader,
                'lagger': lagger,
                'n_windows_significant': n_significant,
                's1_persistence': s1,
                's2_lag_stability': s2,
                's3_sign_consistency': s3,
                's4_strength_stability': s4,
                'stability_score': stability_score,
                'avg_auc': group['auc'].mean(),
                'avg_optimal_lag': group['optimal_lag'].mean(),
                'avg_pearson_corr': group['pearson_corr'].mean()
            })
        
        stability_df = pd.DataFrame(stability_results)
        
        # Filter by stability threshold
        stable_pairs = stability_df[stability_df['stability_score'] >= self.config['stability_threshold']]
        
        print(f"\n✓ Pairs after stability filtering: {len(stable_pairs)}/{len(stability_df)}")
        print(f"\nTop 10 most stable pairs:")
        print(stable_pairs.nlargest(10, 'stability_score')[['leader', 'lagger', 'stability_score',
                                                            'avg_auc', 'avg_optimal_lag']])
        
        self.results['stability'] = stable_pairs
        return stable_pairs
    
    def validate_out_of_sample(self):
        """Step 6: Out-of-sample validation"""
        print("\n" + "=" * 80)
        print("STEP 6: OUT-OF-SAMPLE VALIDATION")
        print("=" * 80)
        
        # Split data
        train_end_date = pd.to_datetime(self.config['train_end'])
        test_start_date = pd.to_datetime(self.config['test_start'])
        
        returns_train = self.data['returns'][self.data['returns'].index <= train_end_date]
        returns_test = self.data['returns'][self.data['returns'].index >= test_start_date]
        
        print(f"\nTraining period: {returns_train.index[0].date()} to {returns_train.index[-1].date()}")
        print(f"Testing period: {returns_test.index[0].date()} to {returns_test.index[-1].date()}")
        
        # Get stable pairs from training period
        stable_pairs = self.results['stability']
        
        # Validate each pair in test period
        validation_results = []
        
        for _, pair in tqdm(stable_pairs.iterrows(), total=len(stable_pairs),
                          desc="Validating pairs"):
            leader = pair['leader']
            lagger = pair['lagger']
            
            if leader not in returns_test.columns or lagger not in returns_test.columns:
                continue
            
            # Compute residuals for test period
            market_returns_test = self.data['market_returns'][
                self.data['market_returns'].index >= test_start_date
            ]
            
            # Leader residuals
            y_leader = returns_test[leader].values
            X = add_constant(market_returns_test.values)
            valid_idx = ~(np.isnan(y_leader) | np.isnan(market_returns_test.values))
            
            if valid_idx.sum() < 100:
                continue
            
            try:
                model_leader = OLS(y_leader[valid_idx], X[valid_idx]).fit()
                resid_leader = y_leader - model_leader.predict(X)
                z_leader = (resid_leader - np.nanmean(resid_leader)) / np.nanstd(resid_leader)
                
                # Lagger residuals
                y_lagger = returns_test[lagger].values
                model_lagger = OLS(y_lagger[valid_idx], X[valid_idx]).fit()
                resid_lagger = y_lagger - model_lagger.predict(X)
                z_lagger = (resid_lagger - np.nanmean(resid_lagger)) / np.nanstd(resid_lagger)
                
                # Test at optimal lag from training
                optimal_lag = int(pair['avg_optimal_lag'])
                
                # Compute test metrics
                dcor_test = self.compute_distance_correlation(z_leader, z_lagger, optimal_lag)
                pearson_test, pearson_pval = self.compute_pearson_at_lag(z_leader, z_lagger, optimal_lag)
                
                # Directional accuracy (does the relationship hold?)
                train_sign = np.sign(pair['avg_pearson_corr'])
                test_sign = np.sign(pearson_test)
                directional_accuracy = 1 if train_sign == test_sign else 0
                
                validation_results.append({
                    'leader': leader,
                    'lagger': lagger,
                    'train_auc': pair['avg_auc'],
                    'test_dcor': dcor_test,
                    'train_corr': pair['avg_pearson_corr'],
                    'test_corr': pearson_test,
                    'test_pval': pearson_pval,
                    'directional_accuracy': directional_accuracy,
                    'optimal_lag': optimal_lag
                })
            except:
                continue
        
        validation_df = pd.DataFrame(validation_results)
        
        # Keep only pairs that validate
        validated_pairs = validation_df[
            (validation_df['test_pval'] < 0.05) &
            (validation_df['directional_accuracy'] == 1) &
            (validation_df['test_dcor'] > 0.1)
        ]
        
        print(f"\n✓ Validated pairs: {len(validated_pairs)}/{len(validation_df)}")
        print(f"  Average test correlation: {validated_pairs['test_corr'].mean():.3f}")
        print(f"  Directional accuracy: {validation_df['directional_accuracy'].mean():.2%}")
        
        self.results['validation'] = validated_pairs
        return validated_pairs
    
    def build_ml_ranking_model(self):
        """Step 8: Machine learning ranking model"""
        print("\n" + "=" * 80)
        print("STEP 8: ML RANKING MODEL (XGBoost)")
        print("=" * 80)
        
        # Prepare dataset
        pair_results = self.results['pair_results']
        stability = self.results['stability']
        
        # Merge data
        ml_data = pair_results.merge(
            stability[['leader', 'lagger', 'stability_score']],
            on=['leader', 'lagger'],
            how='left'
        )
        
        # Feature engineering
        features = []
        
        print("\nEngineering features...")
        for _, row in tqdm(ml_data.iterrows(), total=len(ml_data)):
            leader = row['leader']
            lagger = row['lagger']
            
            # Get market caps and other features (mock data for now)
            # In production, these would come from your BigQuery database
            feature_dict = {
                'auc': row['auc'],
                'optimal_lag': row['optimal_lag'],
                'pearson_corr': row['pearson_corr'],
                'pearson_corr_abs': abs(row['pearson_corr']),
                'stability_score': row.get('stability_score', 0),
                'dcor_mean': np.mean(row['dcor_values']),
                'dcor_std': np.std(row['dcor_values']),
                'dcor_max': np.max(row['dcor_values']),
            }
            features.append(feature_dict)
        
        X = pd.DataFrame(features)
        y = (ml_data['stability_score'] > self.config['stability_threshold']).astype(int)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Positive class ratio: {y_train.mean():.2%}")
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            eval_metric='auc'
        )
        
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 verbose=False)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✓ Model trained successfully")
        print(f"  AUC-ROC: {auc_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop feature importances:")
        print(feature_importance.head(10))
        
        self.results['ml_model'] = model
        self.results['ml_features'] = X.columns.tolist()
        self.results['ml_importance'] = feature_importance
        
        return model, auc_score
    
    def export_results(self):
        """Step 7: Export results"""
        print("\n" + "=" * 80)
        print("STEP 7: EXPORTING RESULTS")
        print("=" * 80)
        
        # Get top N validated pairs
        validated = self.results['validation'].copy()
        validated = validated.merge(
            self.results['stability'][['leader', 'lagger', 'stability_score']],
            on=['leader', 'lagger'],
            how='left'
        )
        
        top_pairs = validated.nlargest(100, 'stability_score')
        
        # Export to CSV
        output_file = self.output_dir / 'top_lead_lag_pairs.csv'
        top_pairs.to_csv(output_file, index=False)
        print(f"\n✓ Exported top pairs to: {output_file}")
        
        # Export full results
        self.results['pair_results'].to_csv(
            self.output_dir / 'all_pair_results.csv', index=False
        )
        self.results['stability'].to_csv(
            self.output_dir / 'stability_metrics.csv', index=False
        )
        
        # Save model
        with open(self.output_dir / 'ml_model.pkl', 'wb') as f:
            pickle.dump(self.results['ml_model'], f)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        print(f"✓ All results saved to: {self.output_dir}")
        
        return top_pairs
    
    def plot_stability_metrics(self):
        """Visualization: Stability metrics"""
        print("\nGenerating stability metrics plot...")
        
        stability = self.results['stability']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # S1: Persistence
        axes[0, 0].hist(stability['s1_persistence'], bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('Significance Persistence (S1)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Significance Persistence')
        axes[0, 0].axvline(stability['s1_persistence'].mean(), color='red',
                          linestyle='--', label=f'Mean: {stability["s1_persistence"].mean():.2f}')
        axes[0, 0].legend()
        
        # S2: Lag Stability
        axes[0, 1].hist(stability['s2_lag_stability'], bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Lag Stability (S2)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Lag Stability')
        axes[0, 1].axvline(stability['s2_lag_stability'].mean(), color='red',
                          linestyle='--', label=f'Mean: {stability["s2_lag_stability"].mean():.2f}')
        axes[0, 1].legend()
        
        # S3: Sign Consistency
        axes[1, 0].hist(stability['s3_sign_consistency'], bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Sign Consistency (S3)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Sign Consistency')
        axes[1, 0].axvline(stability['s3_sign_consistency'].mean(), color='red',
                          linestyle='--', label=f'Mean: {stability["s3_sign_consistency"].mean():.2f}')
        axes[1, 0].legend()
        
        # Overall Stability Score
        axes[1, 1].scatter(stability['avg_auc'], stability['stability_score'],
                          alpha=0.5, s=50)
        axes[1, 1].set_xlabel('Average AUC')
        axes[1, 1].set_ylabel('Stability Score')
        axes[1, 1].set_title('Stability Score vs. Average AUC')
        axes[1, 1].axhline(self.config['stability_threshold'], color='red',
                          linestyle='--', label=f'Threshold: {self.config["stability_threshold"]}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stability_metrics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: stability_metrics.png")
        plt.close()
    
    def plot_out_of_sample_performance(self):
        """Visualization: Out-of-sample performance"""
        print("\nGenerating out-of-sample performance plot...")
        
        validation = self.results['validation']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Train vs Test Correlation
        axes[0, 0].scatter(validation['train_corr'], validation['test_corr'],
                          alpha=0.5, s=50)
        axes[0, 0].plot([-1, 1], [-1, 1], 'r--', label='Perfect correlation')
        axes[0, 0].set_xlabel('Training Correlation')
        axes[0, 0].set_ylabel('Test Correlation')
        axes[0, 0].set_title('Training vs. Test Correlation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Train vs Test AUC/dCor
        axes[0, 1].scatter(validation['train_auc'], validation['test_dcor'],
                          alpha=0.5, s=50)
        axes[0, 1].set_xlabel('Training AUC')
        axes[0, 1].set_ylabel('Test Distance Correlation')
        axes[0, 1].set_title('Training AUC vs. Test dCor')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Directional Accuracy Distribution
        accuracy_counts = validation['directional_accuracy'].value_counts()
        axes[1, 0].bar(['Incorrect', 'Correct'], 
                      [accuracy_counts.get(0, 0), accuracy_counts.get(1, 0)],
                      edgecolor='black')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Directional Accuracy: {validation["directional_accuracy"].mean():.1%}')
        
        # Test p-value distribution
        axes[1, 1].hist(validation['test_pval'], bins=30, edgecolor='black')
        axes[1, 1].set_xlabel('Test P-value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Test P-values')
        axes[1, 1].axvline(0.05, color='red', linestyle='--', label='α = 0.05')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'out_of_sample_performance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: out_of_sample_performance.png")
        plt.close()
    
    def plot_case_study(self, leader, lagger):
        """Detailed case study for a specific pair"""
        print(f"\nGenerating case study: {leader} → {lagger}...")
        
        # Get pair data from all windows
        pair_data = self.results['pair_results'][
            (self.results['pair_results']['leader'] == leader) &
            (self.results['pair_results']['lagger'] == lagger)
        ]
        
        if len(pair_data) == 0:
            print(f"  Warning: No data found for {leader} → {lagger}")
            return
        
        # Get optimal lag
        optimal_lag = int(pair_data['optimal_lag'].mean())
        
        # Get returns data
        returns = self.data['returns']
        leader_returns = returns[leader]
        lagger_returns = returns[lagger]
        
        # Compute residuals for visualization
        market_returns = self.data['market_returns']
        
        # Leader residuals
        y_leader = leader_returns.values
        X = add_constant(market_returns.values)
        valid_idx = ~(np.isnan(y_leader) | np.isnan(market_returns.values))
        model_leader = OLS(y_leader[valid_idx], X[valid_idx]).fit()
        
        # Create full residual series
        resid_leader = np.full(len(leader_returns), np.nan)
        resid_leader[valid_idx] = y_leader[valid_idx] - model_leader.predict(X[valid_idx])
        z_leader = (resid_leader - np.nanmean(resid_leader)) / np.nanstd(resid_leader)
        
        # Lagger residuals
        y_lagger = lagger_returns.values
        model_lagger = OLS(y_lagger[valid_idx], X[valid_idx]).fit()
        resid_lagger = np.full(len(lagger_returns), np.nan)
        resid_lagger[valid_idx] = y_lagger[valid_idx] - model_lagger.predict(X[valid_idx])
        z_lagger = (resid_lagger - np.nanmean(resid_lagger)) / np.nanstd(resid_lagger)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Shifted Overlay Chart (main plot)
        ax1 = fig.add_subplot(gs[0:2, :])
        
        dates = returns.index
        ax1.plot(dates, z_leader, label=f'{leader} (Leader)', alpha=0.7, linewidth=1)
        
        # Shift lagger by optimal lag
        z_lagger_shifted = np.roll(z_lagger, -optimal_lag)
        z_lagger_shifted[-optimal_lag:] = np.nan  # Remove wrapped values
        
        ax1.plot(dates, z_lagger_shifted, label=f'{lagger} (Lagger, shifted by {optimal_lag} days)',
                alpha=0.7, linewidth=1)
        
        # Highlight windows where relationship was significant
        for _, window in pair_data.iterrows():
            ax1.axvspan(window['start_date'], window['end_date'],
                       alpha=0.2, color='green', label='Significant window' if _ == 0 else '')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Standardized Residual Returns')
        ax1.set_title(f'Lead-Lag Relationship: {leader} → {lagger} (Lag: {optimal_lag} days)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cross-correlation bar plot
        ax2 = fig.add_subplot(gs[2, 0])
        
        # Compute cross-correlation for all lags
        lags = self.config['lags']
        dcor_values = []
        for lag in lags:
            dcor_val = self.compute_distance_correlation(z_leader, z_lagger, lag)
            dcor_values.append(dcor_val)
        
        colors = ['red' if lag == optimal_lag else 'steelblue' for lag in lags]
        ax2.bar(lags, dcor_values, color=colors, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Lag (days)')
        ax2.set_ylabel('Distance Correlation')
        ax2.set_title('Cross-Correlation Across Lags')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (lag, val) in enumerate(zip(lags, dcor_values)):
            ax2.text(lag, val, f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top')
        
        # 3. Relationship statistics
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.axis('off')
        
        # Compute statistics
        stats_text = f"""
PAIR STATISTICS

Leader: {leader}
Lagger: {lagger}

Optimal Lag: {optimal_lag} days
Average AUC: {pair_data['auc'].mean():.4f}
Average Correlation: {pair_data['pearson_corr'].mean():.3f}

Significance:
  # Significant Windows: {len(pair_data)}
  Average p-value: {pair_data['perm_pval'].mean():.4f}

Stability (if available):
"""
        
        # Add stability metrics if available
        stability_row = self.results['stability'][
            (self.results['stability']['leader'] == leader) &
            (self.results['stability']['lagger'] == lagger)
        ]
        
        if len(stability_row) > 0:
            stats_text += f"  Persistence (S1): {stability_row['s1_persistence'].values[0]:.3f}\n"
            stats_text += f"  Lag Stability (S2): {stability_row['s2_lag_stability'].values[0]:.3f}\n"
            stats_text += f"  Sign Consistency (S3): {stability_row['s3_sign_consistency'].values[0]:.3f}\n"
            stats_text += f"  Overall Score: {stability_row['stability_score'].values[0]:.3f}\n"
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.savefig(self.output_dir / f'case_study_{leader}_{lagger}.png',
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved: case_study_{leader}_{lagger}.png")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Stability metrics
        self.plot_stability_metrics()
        
        # Out-of-sample performance
        self.plot_out_of_sample_performance()
        
        # Case studies for top 3 pairs
        validated = self.results['validation'].copy()
        validated = validated.merge(
            self.results['stability'][['leader', 'lagger', 'stability_score']],
            on=['leader', 'lagger'],
            how='left'
        )
        
        top_3_pairs = validated.nlargest(3, 'stability_score')
        
        print(f"\nGenerating case studies for top 3 pairs:")
        for _, pair in top_3_pairs.iterrows():
            print(f"  {pair['leader']} → {pair['lagger']}")
            self.plot_case_study(pair['leader'], pair['lagger'])
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}")
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        print("\n" + "=" * 80)
        print("LEAD-LAG DETECTION FRAMEWORK")
        print("Dynamic Rolling Window Version")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Stocks: {self.config['n_stocks']}")
        print(f"  Window: {self.config['window_years']} years")
        print(f"  Step: {self.config['step_years']} year(s)")
        print(f"  Lags: {self.config['lags']}")
        print(f"  Date range: {self.config['start_date']} to {self.config['end_date']}")
        
        # Execute pipeline
        self.download_data()
        self.filter_and_prepare()
        self.create_rolling_windows()
        self.detect_lead_lag_pairs()
        self.compute_stability_metrics()
        self.validate_out_of_sample()
        self.build_ml_ranking_model()
        top_pairs = self.export_results()
        self.generate_all_visualizations()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nTop 10 validated lead-lag pairs:")
        print(top_pairs.head(10)[['leader', 'lagger', 'optimal_lag',
                                   'test_corr', 'stability_score']])
        
        print(f"\n✓ All results saved to: {self.output_dir}")
        print(f"✓ Check the directory for CSV files and visualizations")
        
        return top_pairs


def main():
    """Main execution function"""
    # Initialize framework
    framework = LeadLagFramework(CONFIG)
    
    # Run complete pipeline
    results = framework.run_full_pipeline()
    
    return framework, results


if __name__ == "__main__":
    framework, results = main()
    print("\n" + "=" * 80)
    print("Framework execution complete!")
    print("Access results via: framework.results")
    print("=" * 80)