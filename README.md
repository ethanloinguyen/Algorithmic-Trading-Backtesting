# S&P 500 Correlation Analysis Test

Test script to validate correlation analysis approach before building full Russell 2000 application.

## What This Does

1. Fetches historical stock data from Tiingo API (S&P 500 sample)
2. Calculates daily returns
3. Computes correlation matrix (NOT covariance as correlation is bounded between -1 to 1)
4. Generates interactive Plotly heatmap
5. Saves all data to CSV files (will save to BigQuery once connected)

## Prerequisites

1. **Tiingo API Key** (Free tier works up to 500 unique tickers, need paid pro tier for more tickers)
   - Available on https://www.tiingo.com/

2. **Python 3.8+**

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit sp500_correlation_test.py
# Replace line 15 with your API key:
TIINGO_API_KEY = "your_actual_api_key_here"
```

## Usage

### Run the full test:
```bash
python sp500_correlation_test.py
```

This will:
- Fetch 2 years of data for 20 S&P 500 stocks
- Calculate correlation matrix
- Generate interactive heatmap
- Save everything to `sp500_data/` folder

### Viewing the heatmap:
```bash
# Open the generated file in your browser
open sp500_data/correlation_heatmap.html

# Or manually navigate to the file
```

### Generate heatmap from existing data:
```bash
python plotly_heatmap.py sp500_data/correlation_matrix.csv
```

## Output Files

```
sp500_data/
├── price_data.csv              # Raw adjusted close prices
├── returns.csv                 # Daily percentage returns
├── correlation_matrix.csv      # Correlation coefficients
└── correlation_heatmap.html    # Interactive visualization
```

## Interpreting Results

### Correlation Values:
- **1.0**: Perfect positive correlation (stocks move exactly together)
- **-0.2 to 0.2**: Weak/no correlation (good for diversification!)
- **-1.0**: Perfect negative correlation (stocks move exactly opposite)

## Customization

### How to test more stocks:
```python
# In sp500_correlation_test.py, line 18-23
# Add more tickers to SP500_TICKERS list
SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    # ... add more ...
    'WMT', 'KO', 'PEP', 'T', 'VZ'
]
```

### Change date range:
```python
# Line 14
START_DATE = "2020-01-01"  # 5 years of data (can change to expand or close scope)
```

### Test specific sectors:
```python
# Can later assign sectors/industries to each stock ticker when BigQuery is connected
# Technology stocks
TECH_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']

# Financial stocks
FINANCE_STOCKS = ['JPM', 'BAC', 'WFC', 'GS', 'MS']

# Healthcare stocks
HEALTHCARE_STOCKS = ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO']
```

## Troubleshooting

### "No data returned" errors:
- Check your API key is correct
- Verify ticker symbols are valid
- Free tier has 500 unique symbols/month limit

### Rate limiting errors:
- Script includes 1.2 second delays between requests
- For more stocks, increase `time.sleep(1.2)` to `time.sleep(2)`

### ImportError for plotly:
```bash
pip install plotly
```