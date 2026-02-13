#!/usr/bin/env python3
"""
Lead-Lag Analysis Script
Analyzes time-delayed correlations between stocks to identify leaders and laggers
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load environment variables
load_dotenv()

# Configuration
TIINGO_API_KEY = os.getenv('TIINGO_API_TOKEN')
OUTPUT_DIR = "lead_lag_results"
START_DATE = "2023-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')

# S&P 500 sample tickers (same as before heatmap tickers)
SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA',
    'XOM', 'BAC', 'ABBV', 'COST', 'DIS'
]

def setup_directories():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ready: {OUTPUT_DIR}/")

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Tiingo API
    """
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'token': TIINGO_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print(f"{ticker}: No data returned")
            return None
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df[['date', 'adjClose']].rename(columns={'adjClose': ticker})
        df.set_index('date', inplace=True)
        
        print(f"{ticker}: {len(df)} days fetched")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"{ticker}: Error - {e}")
        return None

def fetch_all_tickers(tickers, start_date, end_date):
    """
    Fetch data for all tickers with rate limiting
    """
    print(f"\n{'='*60}")
    print(f"Fetching data for {len(tickers)} stocks")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    all_data = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Fetching {ticker}...")
        
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            all_data.append(df)
        
        time.sleep(1.2)  # Rate limiting
    
    if not all_data:
        print("\nNo data fetched!")
        return None
    
    combined = pd.concat(all_data, axis=1, join='inner')
    
    print(f"\n{'='*60}")
    print(f"  Data collection complete!")
    print(f"  Total tickers: {len(combined.columns)}")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
    print(f"  Trading days: {len(combined)}")
    print(f"{'='*60}\n")
    
    return combined

def calculate_returns(price_data):
    """Calculate daily returns from price data"""
    returns = price_data.pct_change().dropna()
    print(f"Calculated daily returns ({len(returns)} days)")
    return returns

# ============================================================================
# LEAD-LAG ANALYSIS FUNCTIONS
# ============================================================================

def calculate_lagged_correlation(stock_a, stock_b, returns_df, max_lag=5):
    """
    Calculate correlation between stock_a and stock_b at different lags
    
    Args:
        stock_a: ticker symbol (e.g., 'AAPL')
        stock_b: ticker symbol (e.g., 'MSFT')
        returns_df: DataFrame with daily returns
        max_lag: maximum days to check (default 5)
    
    Returns:
        dict: {lag: correlation} for each lag from -max_lag to +max_lag
        
    Interpretation:
        - lag > 0: stock_a TODAY predicts stock_b in FUTURE (stock_a leads)
        - lag < 0: stock_b TODAY predicts stock_a in FUTURE (stock_b leads)
        - lag = 0: stocks move together synchronously
    """
    results = {}
    
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            # Same-day correlation (synchronous)
            results[0] = returns_df[stock_a].corr(returns_df[stock_b])
        elif lag > 0:
            # Positive lag: stock_a TODAY predicts stock_b FUTURE
            a_series = returns_df[stock_a].iloc[:-lag]
            b_series = returns_df[stock_b].iloc[lag:]
            results[lag] = a_series.corr(b_series)
        else:  # lag < 0
            # Negative lag: stock_b TODAY predicts stock_a FUTURE
            lag_abs = abs(lag)
            b_series = returns_df[stock_b].iloc[:-lag_abs]
            a_series = returns_df[stock_a].iloc[lag_abs:]
            results[lag] = a_series.corr(b_series)
    
    return results

def find_best_lag(stock_a, stock_b, returns_df, max_lag=5):
    """
    Find the lag with highest absolute correlation
    
    Returns:
        tuple: (best_lag, correlation, interpretation)
    """
    lag_corrs = calculate_lagged_correlation(stock_a, stock_b, returns_df, max_lag)
    
    # Find lag with maximum absolute correlation
    best_lag = max(lag_corrs.items(), key=lambda x: abs(x[1]))
    lag_days = best_lag[0]
    correlation = best_lag[1]
    
    # Interpret the result
    if lag_days > 0:
        interpretation = f"{stock_a} leads {stock_b} by {lag_days} days"
    elif lag_days < 0:
        interpretation = f"{stock_b} leads {stock_a} by {abs(lag_days)} days"
    else:
        interpretation = f"{stock_a} and {stock_b} move synchronously"
    
    return lag_days, correlation, interpretation

def analyze_all_pairs(returns_df, max_lag=5):
    """
    Analyze lead-lag relationships for all stock pairs
    
    Returns:
        DataFrame with columns: Stock_A, Stock_B, Best_Lag, Correlation, Leader, Lagger
    """
    print(f"\n{'='*60}")
    print("Analyzing all pairwise lead-lag relationships...")
    print(f"{'='*60}\n")
    
    results = []
    tickers = returns_df.columns.tolist()
    total_pairs = len(tickers) * (len(tickers) - 1) // 2
    
    print(f"Total pairs to analyze: {total_pairs}")
    
    for i, stock_a in enumerate(tickers):
        for stock_b in tickers[i+1:]:
            lag, corr, interp = find_best_lag(stock_a, stock_b, returns_df, max_lag)
            
            # Determine leader
            if lag > 0:
                leader = stock_a
                lagger = stock_b
            elif lag < 0:
                leader = stock_b
                lagger = stock_a
                lag = abs(lag)
            else:
                leader = "Synchronous"
                lagger = "Synchronous"
            
            results.append({
                'Stock_A': stock_a,
                'Stock_B': stock_b,
                'Best_Lag': lag,
                'Correlation': abs(corr),
                'Signed_Correlation': corr,
                'Leader': leader,
                'Lagger': lagger,
                'Interpretation': interp
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Correlation', ascending=False)
    
    print(f"Analysis complete: {len(df)} pairs analyzed\n")
    
    return df

def find_top_leaders(lag_analysis_df, top_n=10):
    """
    Find stocks that lead the most other stocks
    
    Returns:
        DataFrame with leader rankings
    """
    # Filter out synchronous relationships
    leading_relationships = lag_analysis_df[lag_analysis_df['Leader'] != 'Synchronous']
    
    # Count how many stocks each ticker leads
    leader_counts = leading_relationships['Leader'].value_counts()
    
    # Calculate average correlation when leading
    avg_corr = leading_relationships.groupby('Leader')['Correlation'].mean()
    
    # Calculate average lag when leading
    avg_lag = leading_relationships.groupby('Leader')['Best_Lag'].mean()
    
    leaders = pd.DataFrame({
        'Stock': leader_counts.index,
        'Times_as_Leader': leader_counts.values,
        'Avg_Correlation': [avg_corr[stock] for stock in leader_counts.index],
        'Avg_Lag_Days': [avg_lag[stock] for stock in leader_counts.index]
    }).head(top_n)
    
    return leaders

def find_top_laggers(lag_analysis_df, top_n=10):
    """
    Find stocks that lag behind the most other stocks
    
    Returns:
        DataFrame with lagger rankings
    """
    # Filter out synchronous relationships
    lagging_relationships = lag_analysis_df[lag_analysis_df['Lagger'] != 'Synchronous']
    
    # Count how many stocks each ticker lags behind
    lagger_counts = lagging_relationships['Lagger'].value_counts()
    
    # Calculate average correlation when lagging
    avg_corr = lagging_relationships.groupby('Lagger')['Correlation'].mean()
    
    # Calculate average lag when lagging
    avg_lag = lagging_relationships.groupby('Lagger')['Best_Lag'].mean()
    
    laggers = pd.DataFrame({
        'Stock': lagger_counts.index,
        'Times_as_Lagger': lagger_counts.values,
        'Avg_Correlation': [avg_corr[stock] for stock in lagger_counts.index],
        'Avg_Lag_Days': [avg_lag[stock] for stock in lagger_counts.index]
    }).head(top_n)
    
    return laggers

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_lag_correlation(stock_a, stock_b, returns_df, max_lag=5):
    """
    Plot correlation vs lag for two stocks
    """
    lag_corrs = calculate_lagged_correlation(stock_a, stock_b, returns_df, max_lag)
    
    lags = list(lag_corrs.keys())
    correlations = list(lag_corrs.values())
    
    # Find best lag
    best_idx = np.argmax([abs(c) for c in correlations])
    best_lag = lags[best_idx]
    best_corr = correlations[best_idx]
    
    fig = go.Figure()
    
    # Add bar chart
    colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in lags]
    colors[best_idx] = 'gold'
    
    fig.add_trace(go.Bar(
        x=lags,
        y=correlations,
        marker_color=colors,
        text=[f'{c:.3f}' for c in correlations],
        textposition='outside',
        name='Correlation'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f'Lead-Lag Correlation: {stock_a} vs {stock_b}<br><sub>Best: {best_lag} days (r={best_corr:.3f})</sub>',
        xaxis_title='Lag (days)',
        yaxis_title='Correlation',
        xaxis=dict(tickmode='linear', tick0=-max_lag, dtick=1),
        showlegend=False,
        annotations=[
            dict(
                x=0.5, y=1.15,
                xref='paper', yref='paper',
                text=f'Negative lag: {stock_b} leads | Positive lag: {stock_a} leads',
                showarrow=False,
                font=dict(size=10, color='gray')
            )
        ],
        height=500
    )
    
    return fig

def plot_time_series_comparison(stock_a, stock_b, price_data):
    """
    Plot normalized price movements of two stocks
    """
    # Normalize prices to start at 100
    norm_a = (price_data[stock_a] / price_data[stock_a].iloc[0]) * 100
    norm_b = (price_data[stock_b] / price_data[stock_b].iloc[0]) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=norm_a,
        name=stock_a,
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=norm_b,
        name=stock_b,
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'Price Movement Comparison: {stock_a} vs {stock_b}',
        xaxis_title='Date',
        yaxis_title='Normalized Price (Base = 100)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_comprehensive_analysis(stock_a, stock_b, returns_df, price_data, max_lag=5):
    """
    Create comprehensive 2-panel visualization showing lag analysis and price comparison
    """
    # Get lag correlations
    lag_corrs = calculate_lagged_correlation(stock_a, stock_b, returns_df, max_lag)
    lags = list(lag_corrs.keys())
    correlations = list(lag_corrs.values())
    
    # Find best lag
    best_idx = np.argmax([abs(c) for c in correlations])
    best_lag = lags[best_idx]
    best_corr = correlations[best_idx]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Lag-Correlation Analysis',
            f'Price Movement Comparison'
        ),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.6]
    )
    
    # Top plot: Lag-correlation bar chart
    colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in lags]
    colors[best_idx] = 'gold'
    
    fig.add_trace(
        go.Bar(
            x=lags,
            y=correlations,
            marker_color=colors,
            text=[f'{c:.3f}' for c in correlations],
            textposition='outside',
            name='Correlation',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add zero line to top plot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Bottom plot: Time series comparison
    norm_a = (price_data[stock_a] / price_data[stock_a].iloc[0]) * 100
    norm_b = (price_data[stock_b] / price_data[stock_b].iloc[0]) * 100
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=norm_a,
            name=stock_a,
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=norm_b,
            name=stock_b,
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Update axes
    fig.update_xaxes(title_text="Lag (days)", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Price (Base=100)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text=f"Lead-Lag Analysis: {stock_a} vs {stock_b}<br><sub>Best Lag: {best_lag} days (Correlation: {best_corr:.3f})</sub>",
        hovermode='x unified'
    )
    
    # Add annotation
    interpretation = ""
    if best_lag > 0:
        interpretation = f"{stock_a} leads {stock_b} by {best_lag} days"
    elif best_lag < 0:
        interpretation = f"{stock_b} leads {stock_a} by {abs(best_lag)} days"
    else:
        interpretation = f"{stock_a} and {stock_b} move synchronously"
    
    fig.add_annotation(
        text=interpretation,
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        showarrow=False,
        font=dict(size=12, color="green"),
        xanchor='center'
    )
    
    return fig

def plot_leader_rankings(leaders_df):
    """
    Bar chart showing top leading stocks
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=leaders_df['Stock'],
        y=leaders_df['Times_as_Leader'],
        text=leaders_df['Times_as_Leader'],
        textposition='outside',
        marker_color='lightblue',
        name='Times as Leader'
    ))
    
    fig.update_layout(
        title='Top Leading Stocks<br><sub>Number of stocks they lead</sub>',
        xaxis_title='Stock',
        yaxis_title='Number of Stocks Led',
        showlegend=False,
        height=500
    )
    
    return fig

def plot_lagger_rankings(laggers_df):
    """
    Bar chart showing top lagging stocks
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=laggers_df['Stock'],
        y=laggers_df['Times_as_Lagger'],
        text=laggers_df['Times_as_Lagger'],
        textposition='outside',
        marker_color='lightcoral',
        name='Times as Lagger'
    ))
    
    fig.update_layout(
        title='Top Lagging Stocks<br><sub>Number of stocks they lag behind</sub>',
        xaxis_title='Stock',
        yaxis_title='Number of Stocks Lagged Behind',
        showlegend=False,
        height=500
    )
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("LEAD-LAG ANALYSIS")
    print("="*60 + "\n")
    
    # Check API key
    if not TIINGO_API_KEY or TIINGO_API_KEY == "YOUR_API_KEY_HERE":
        print("Error: Tiingo API key not found!")
        print("Create a .env file with: TIINGO_API_TOKEN=your_key_here")
        return
    
    # Setup
    setup_directories()
    
    # Step 1: Fetch data
    print("Step 1: Fetching stock data...")
    price_data = fetch_all_tickers(SP500_TICKERS, START_DATE, END_DATE)
    
    if price_data is None or price_data.empty:
        print("No data to process. Exiting.")
        return
    
    # Step 2: Calculate returns
    print("\nStep 2: Calculating returns...")
    returns = calculate_returns(price_data)
    
    # Step 3: Analyze all pairs
    print("\nStep 3: Analyzing lead-lag relationships...")
    lag_analysis = analyze_all_pairs(returns, max_lag=5)
    
    # Save results
    lag_analysis.to_csv(f"{OUTPUT_DIR}/all_pairs_analysis.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR}/all_pairs_analysis.csv\n")
    
    # Step 4: Find top leaders and laggers
    print("\nStep 4: Identifying top leaders and laggers...")
    
    top_leaders = find_top_leaders(lag_analysis, top_n=10)
    top_laggers = find_top_laggers(lag_analysis, top_n=10)
    
    print("\n" + "="*60)
    print("TOP LEADING STOCKS:")
    print("="*60)
    print(top_leaders.to_string(index=False))
    
    print("\n" + "="*60)
    print("TOP LAGGING STOCKS:")
    print("="*60)
    print(top_laggers.to_string(index=False))
    print("="*60 + "\n")
    
    # Save rankings
    top_leaders.to_csv(f"{OUTPUT_DIR}/top_leaders.csv", index=False)
    top_laggers.to_csv(f"{OUTPUT_DIR}/top_laggers.csv", index=False)
    
    # Step 5: Generate visualizations
    print("\nStep 5: Generating visualizations...")
    
    # Example pair analysis (AAPL vs MSFT)
    if 'AAPL' in returns.columns and 'MSFT' in returns.columns:
        fig1 = plot_comprehensive_analysis('AAPL', 'MSFT', returns, price_data, max_lag=5)
        fig1.write_html(f"{OUTPUT_DIR}/aapl_msft_analysis.html")
        print(f"Saved: {OUTPUT_DIR}/aapl_msft_analysis.html")
    
    # Leader rankings chart
    fig2 = plot_leader_rankings(top_leaders)
    fig2.write_html(f"{OUTPUT_DIR}/leader_rankings.html")
    print(f"Saved: {OUTPUT_DIR}/leader_rankings.html")
    
    # Lagger rankings chart
    fig3 = plot_lagger_rankings(top_laggers)
    fig3.write_html(f"{OUTPUT_DIR}/lagger_rankings.html")
    print(f"Saved: {OUTPUT_DIR}/lagger_rankings.html")
    
    # Step 6: Show strongest relationships
    print("\n" + "="*60)
    print("STRONGEST LEAD-LAG RELATIONSHIPS:")
    print("="*60)
    print(lag_analysis.head(10)[['Stock_A', 'Stock_B', 'Best_Lag', 'Correlation', 'Interpretation']].to_string(index=False))
    print("="*60 + "\n")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60 + "\n")
    
    print("Files generated:")
    print(f"  - {OUTPUT_DIR}/all_pairs_analysis.csv")
    print(f"  - {OUTPUT_DIR}/top_leaders.csv")
    print(f"  - {OUTPUT_DIR}/top_laggers.csv")
    print(f"  - {OUTPUT_DIR}/aapl_msft_analysis.html")
    print(f"  - {OUTPUT_DIR}/leader_rankings.html")
    print(f"  - {OUTPUT_DIR}/lagger_rankings.html")
    
    print("\nNext steps:")
    print("1. Open the HTML files in your browser")
    print("2. Review the top leaders and laggers")
    print("3. Examine the strongest lead-lag relationships")
    print("4. Consider scaling to full Russell 2000 dataset")

if __name__ == "__main__":
    main()