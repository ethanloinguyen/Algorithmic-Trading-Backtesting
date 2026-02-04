# S&P 500 Correlation Analysis Test

Test script to validate correlation analysis approach before building full Russell 2000 application.

## What This Does

1. Fetches historical stock data from Tiingo API (S&P 500 sample)
2. Calculates daily returns
3. Computes correlation matrix (NOT covariance - correlation is bounded -1 to 1)
4. Generates interactive Plotly heatmap
5. Saves all data to CSV files

## Prerequisites

1. **Tiingo API Key** (Free tier works!)
   - Sign up at https://www.tiingo.com/
   - Go to Account → API
   - Copy your token

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

### View the heatmap:
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
- **0.5 to 1.0**: Strong positive correlation
- **0.2 to 0.5**: Moderate positive correlation
- **-0.2 to 0.2**: Weak/no correlation (good for diversification!)
- **-0.5 to -0.2**: Moderate negative correlation
- **-1.0 to -0.5**: Strong negative correlation
- **-1.0**: Perfect negative correlation (stocks move exactly opposite)

### What to Look For:
✅ **Good signs** (your project is valuable):
- Average correlation < 0.7
- Multiple pairs with correlation < 0.3
- Some negative correlations
- Variation between sectors

❌ **Bad signs** (might need to pivot):
- Average correlation > 0.9
- All correlations > 0.8
- No variation between stocks

## Customization

### Test more stocks:
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
START_DATE = "2020-01-01"  # 5 years of data
```

### Test specific sectors:
```python
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

## Next Steps After Testing

If the test shows meaningful correlation variation:

1. ✅ **Proceed with full project**
   - Upgrade to Tiingo paid tier ($30/month)
   - Implement BigQuery storage
   - Build Next.js frontend
   - Deploy to Google Cloud

2. **Add features:**
   - Sector/industry filtering
   - Variable matrix sizes (5×5, 10×10, 25×25)
   - "Find diversifiers" recommendation engine
   - Time-period comparison

3. **Architecture:**
   ```
   Cloud Scheduler → Cloud Functions → BigQuery
                                     ↓
                               Correlation matrices
                                     ↓
                     Cloud Run (FastAPI) ← Next.js Frontend
   ```

## Key Insights from Testing

After running this script, you should be able to answer:
- Is there meaningful variation in correlations? (YES = project valuable)
- What's the typical correlation between stocks? (~0.3-0.7 for diverse stocks)
- Are there negative correlations? (Usually a few)
- How do sectors differ? (Tech stocks highly correlated, Energy less so)

## Questions?

Check the code comments or refer back to the architecture discussion.

Good luck with your capstone project! 🚀