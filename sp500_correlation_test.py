#!/usr/bin/env python3
"""
S&P 500 Correlation Analysis Test Script
Tests the correlation pipeline without BigQuery
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
TIINGO_API_KEY = os.getenv('TIINGO_API_TOKEN')
OUTPUT_DIR = "sp500_data"
START_DATE = "2023-01-01"  # 2 years of data for testing
END_DATE = datetime.now().strftime('%Y-%m-%d')

# S&P 500 sample tickers (top 20 for quick testing)
# For full S&P 500, you'd use all ~503 tickers
# Note: BRK.B is formatted as BRK-B in Tiingo
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
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with date, close prices, or None if error
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
        
        # Extract relevant fields
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
    
    Returns:
        DataFrame with columns = tickers, rows = dates
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
        
        # Rate limiting: Free tier allows ~1 request/sec
        time.sleep(1.2)
    
    if not all_data:
        print("\nNo data fetched!")
        return None
    
    # Combine all dataframes
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

def calculate_correlation_matrix(returns):
    """Calculate correlation matrix from returns"""
    corr_matrix = returns.corr()
    
    print(f"\n{'='*60}")
    print("Correlation Matrix Statistics:")
    print(f"{'='*60}")
    
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlations = corr_matrix.where(mask).stack()
    
    print(f"Average correlation: {correlations.mean():.3f}")
    print(f"Max correlation: {correlations.max():.3f}")
    print(f"Min correlation: {correlations.min():.3f}")
    print(f"Std deviation: {correlations.std():.3f}")
    
    # Find extreme pairs
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Stock1': corr_matrix.columns[i],
                'Stock2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation')
    
    print(f"\nMost negatively correlated:")
    print(corr_df.head(3).to_string(index=False))
    
    print(f"\nMost positively correlated:")
    print(corr_df.tail(3).to_string(index=False))
    
    print(f"{'='*60}\n")
    
    return corr_matrix

def save_data(price_data, returns, corr_matrix):
    """Save all data to CSV files"""
    price_data.to_csv(f"{OUTPUT_DIR}/price_data.csv")
    returns.to_csv(f"{OUTPUT_DIR}/returns.csv")
    corr_matrix.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
    
    print("✓ Data saved:")
    print(f"  - {OUTPUT_DIR}/price_data.csv")
    print(f"  - {OUTPUT_DIR}/returns.csv")
    print(f"  - {OUTPUT_DIR}/correlation_matrix.csv")

def generate_plotly_heatmap(corr_matrix):
    """Generate interactive Plotly heatmap"""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',  # Red-Blue color scheme
        zmid=0,  # Center colorscale at 0
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
        )
    ))
    
    fig.update_layout(
        title={
            'text': f'S&P 500 Stock Correlation Matrix<br><sub>({START_DATE} to {END_DATE})</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={'title': 'Stock Ticker', 'side': 'bottom'},
        yaxis={'title': 'Stock Ticker'},
        width=1000,
        height=900,
        font=dict(size=10)
    )
    
    # Save as HTML
    output_file = f"{OUTPUT_DIR}/correlation_heatmap.html"
    fig.write_html(output_file)
    
    print(f"\n Interactive heatmap saved: {output_file}")
    print("  Open this file in your browser to view the heatmap!")
    
    return fig

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("S&P 500 CORRELATION ANALYSIS TEST")
    print("="*60 + "\n")
    
    # Check API key
    if not TIINGO_API_KEY or TIINGO_API_KEY == "YOUR_API_KEY_HERE":
        print("Error: Tiingo API key not found!")
        print("Create a .env file in the project root with:")
        print("TIINGO_API_TOKEN=your_actual_api_key_here")
        return
    
    # Step 1: Setup
    setup_directories()
    
    # Step 2: Fetch data
    price_data = fetch_all_tickers(SP500_TICKERS, START_DATE, END_DATE)
    
    if price_data is None or price_data.empty:
        print("No data to process. Exiting.")
        return
    
    # Step 3: Calculate returns
    returns = calculate_returns(price_data)
    
    # Step 4: Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(returns)
    
    # Step 5: Save data
    save_data(price_data, returns, corr_matrix)
    
    # Step 6: Generate visualization
    try:
        generate_plotly_heatmap(corr_matrix)
    except ImportError:
        print("\nPlotly not installed. Install with: pip install plotly")
        print("   Heatmap generation skipped.")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("1. Open sp500_data/correlation_heatmap.html in your browser")
    print("2. Examine the correlation patterns")
    print("3. Check if there's meaningful variation (there should be!)")
    print("4. If satisfied, proceed with full Russell 2000 implementation")

if __name__ == "__main__":
    main()