# backend/app/services/mock_final_network.py
"""
Mock data for the final_network BigQuery table.
Schema-identical to production — swap get_mock_final_network() call
in portfolio_service.py with a real BQ query when data is available.
"""
from datetime import date
import pandas as pd

AS_OF = date(2025, 3, 31)

# fmt: off
# Tuple layout (17 elements, indices 0-16):
# 0:ti  1:tj  2:lag  3:mean_dcor  4:var_dcor  5:freq  6:half_life
# 7:sharpness  8:pred_sharpe  9:signal_str  10:oos_sharpe  11:oos_dcor
# 12:sector_i  13:sector_j  14:rank(-1=None)  15:centrality_i  16:centrality_j
_RAW = [
    ("NVDA","AMD",  1,0.341,0.0021,0.82,45.2,0.78,1.42,88.1,1.31,0.29,"Technology","Technology",  1,0.91,0.74),
    ("AAPL","MSFT", 2,0.198,0.0018,0.71,68.4,0.61,0.87,72.3,0.81,0.17,"Technology","Technology",  8,0.85,0.88),
    ("MSFT","GOOGL",1,0.221,0.0024,0.74,55.1,0.69,0.94,74.8,0.88,0.19,"Technology","Technology",  6,0.88,0.82),
    ("NVDA","INTC", 1,0.287,0.0031,0.76,38.7,0.72,1.18,81.4,1.09,0.24,"Technology","Technology",  3,0.91,0.61),
    ("AMD", "INTC", 2,0.254,0.0028,0.69,42.3,0.65,1.02,77.6,0.94,0.21,"Technology","Technology",  5,0.74,0.61),
    ("QCOM","AMD",  1,0.219,0.0019,0.67,51.8,0.63,0.91,73.2,0.84,0.18,"Technology","Technology", -1,0.68,0.74),
    ("GOOGL","META",1,0.231,0.0022,0.73,59.4,0.67,0.96,75.4,0.89,0.20,"Technology","Technology", -1,0.82,0.79),
    ("MSFT","AAPL", 3,0.172,0.0016,0.61,82.1,0.54,0.74,67.8,0.68,0.14,"Technology","Technology", -1,0.88,0.85),
    ("NVDA","GOOGL",2,0.263,0.0027,0.71,44.6,0.70,1.07,79.1,0.99,0.22,"Technology","Technology", -1,0.91,0.82),
    ("AAPL","META", 2,0.184,0.0017,0.65,74.2,0.58,0.79,69.5,0.73,0.15,"Technology","Technology", -1,0.85,0.79),
    ("MSFT","V",    2,0.178,0.0015,0.63,71.3,0.56,0.76,68.4,0.71,0.15,"Technology","Financials",  -1,0.88,0.72),
    ("AAPL","MA",   2,0.169,0.0014,0.61,78.6,0.53,0.72,66.9,0.67,0.14,"Technology","Financials",  -1,0.85,0.69),
    ("GOOGL","V",   1,0.192,0.0018,0.66,63.4,0.60,0.82,70.7,0.76,0.16,"Technology","Financials",  -1,0.82,0.72),
    ("JPM","BAC",   1,0.312,0.0029,0.81,36.4,0.76,1.28,85.3,1.19,0.27,"Financials","Financials",   2,0.87,0.79),
    ("GS", "MS",    1,0.298,0.0026,0.79,39.8,0.74,1.22,83.7,1.13,0.25,"Financials","Financials",  -1,0.83,0.78),
    ("JPM","MS",    2,0.241,0.0023,0.72,52.7,0.66,0.98,76.1,0.91,0.20,"Financials","Financials",  -1,0.87,0.78),
    ("BLK","GS",    1,0.227,0.0021,0.70,57.3,0.64,0.93,74.2,0.86,0.19,"Financials","Financials",  -1,0.76,0.83),
    ("V",  "MA",    1,0.334,0.0019,0.84,41.2,0.81,1.38,87.4,1.28,0.28,"Financials","Financials",  -1,0.72,0.69),
    ("XOM","CVX",   1,0.356,0.0033,0.86,33.1,0.82,1.46,89.2,1.35,0.30,"Energy","Energy",           -1,0.88,0.84),
    ("XOM","COP",   2,0.289,0.0028,0.77,46.8,0.71,1.19,81.9,1.10,0.25,"Energy","Energy",           -1,0.88,0.71),
    ("CVX","SLB",   1,0.261,0.0025,0.74,50.2,0.67,1.07,79.3,0.98,0.22,"Energy","Energy",           -1,0.84,0.66),
    ("COP","SLB",   2,0.234,0.0022,0.69,55.9,0.63,0.96,75.6,0.89,0.20,"Energy","Energy",           -1,0.71,0.66),
    ("XOM","CAT",   3,0.168,0.0016,0.58,84.3,0.52,0.71,66.3,0.66,0.14,"Energy","Industrials",      -1,0.88,0.73),
    ("CVX","GE",    2,0.172,0.0015,0.60,79.7,0.54,0.73,67.1,0.68,0.14,"Energy","Industrials",      -1,0.84,0.67),
    ("JNJ","PFE",   2,0.247,0.0024,0.73,49.6,0.67,1.01,77.2,0.93,0.21,"Healthcare","Healthcare",   -1,0.81,0.74),
    ("UNH","ABBV",  1,0.219,0.0020,0.68,58.4,0.62,0.90,72.8,0.84,0.18,"Healthcare","Healthcare",   -1,0.78,0.69),
    ("MRK","PFE",   1,0.236,0.0022,0.71,53.1,0.65,0.97,75.9,0.90,0.20,"Healthcare","Healthcare",   -1,0.72,0.74),
    ("JNJ","MRK",   2,0.208,0.0019,0.66,62.8,0.59,0.88,71.6,0.82,0.17,"Healthcare","Healthcare",   -1,0.81,0.72),
    ("AMZN","COST", 2,0.223,0.0021,0.69,56.7,0.63,0.92,73.6,0.86,0.19,"Consumer","Consumer",       -1,0.84,0.67),
    ("TSLA","AMZN", 1,0.194,0.0019,0.64,67.4,0.58,0.83,70.2,0.77,0.16,"Consumer","Consumer",       -1,0.76,0.84),
    ("HD",  "COST", 1,0.238,0.0022,0.72,51.4,0.66,0.98,76.3,0.91,0.20,"Consumer","Consumer",       -1,0.69,0.67),
    ("NKE", "HD",   2,0.181,0.0017,0.62,73.9,0.55,0.77,68.9,0.72,0.15,"Consumer","Consumer",       -1,0.64,0.69),
    ("CAT", "HON",  1,0.243,0.0023,0.72,50.8,0.66,0.99,76.7,0.92,0.21,"Industrials","Industrials", -1,0.73,0.70),
    ("GE",  "HON",  2,0.216,0.0020,0.67,59.3,0.61,0.89,72.4,0.83,0.18,"Industrials","Industrials", -1,0.67,0.70),
    ("BA",  "CAT",  1,0.204,0.0019,0.65,63.7,0.58,0.87,71.3,0.81,0.17,"Industrials","Industrials", -1,0.65,0.73),
    ("JPM", "AMZN", 2,0.174,0.0016,0.60,77.2,0.54,0.74,67.5,0.69,0.14,"Financials","Consumer",     -1,0.87,0.84),
    ("NVDA","TSLA", 1,0.271,0.0029,0.73,43.1,0.71,1.11,80.2,1.02,0.23,"Technology","Consumer",      4,0.91,0.76),
    ("XOM", "BA",   3,0.159,0.0015,0.56,88.4,0.50,0.68,64.8,0.63,0.13,"Energy","Industrials",      -1,0.88,0.65),
    ("JPM", "HON",  2,0.163,0.0015,0.57,82.6,0.51,0.70,65.6,0.65,0.13,"Financials","Industrials",  -1,0.87,0.70),
    ("MSFT","UNH",  2,0.176,0.0016,0.62,75.8,0.55,0.75,68.1,0.70,0.15,"Technology","Healthcare",   -1,0.88,0.78),
    ("AAPL","JNJ",  3,0.161,0.0014,0.57,86.2,0.51,0.69,65.2,0.64,0.13,"Technology","Healthcare",   -1,0.85,0.81),
    ("GS",  "XOM",  1,0.188,0.0018,0.64,65.9,0.57,0.80,70.0,0.75,0.16,"Financials","Energy",       -1,0.83,0.88),
    ("NVDA","NEE",  4,0.148,0.0014,0.52,96.3,0.47,0.63,62.4,0.59,0.12,"Technology","Utilities",     -1,0.91,0.58),
    ("JPM", "NEE",  3,0.153,0.0013,0.54,91.7,0.49,0.65,63.1,0.61,0.12,"Financials","Utilities",    -1,0.87,0.58),
    ("UNH", "COST", 2,0.179,0.0017,0.62,74.1,0.55,0.77,68.7,0.71,0.15,"Healthcare","Consumer",     -1,0.78,0.67),
    ("NEE", "DUK",  1,0.318,0.0027,0.83,37.6,0.78,1.31,86.1,1.21,0.27,"Utilities","Utilities",     -1,0.58,0.54),
]
# fmt: on

COLUMNS = [
    "as_of_date","ticker_i","ticker_j","best_lag","mean_dcor","variance_dcor",
    "frequency","half_life","sharpness","predicted_sharpe","signal_strength",
    "oos_sharpe_net","oos_dcor","sector_i","sector_j","rank","centrality_i","centrality_j",
]


def get_mock_final_network() -> pd.DataFrame:
    rows = []
    for r in _RAW:
        rows.append({
            "as_of_date":       AS_OF,
            "ticker_i":         r[0],  "ticker_j":         r[1],
            "best_lag":         r[2],  "mean_dcor":        r[3],
            "variance_dcor":    r[4],  "frequency":        r[5],
            "half_life":        r[6],  "sharpness":        r[7],
            "predicted_sharpe": r[8],  "signal_strength":  r[9],
            "oos_sharpe_net":   r[10], "oos_dcor":         r[11],
            "sector_i":         r[12], "sector_j":         r[13],
            "rank":             r[14] if r[14] != -1 else None,
            "centrality_i":     r[15], "centrality_j":     r[16],
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def get_ticker_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ticker → (sector, centrality) from the network DataFrame."""
    side_i = df[["ticker_i","sector_i","centrality_i"]].rename(
        columns={"ticker_i":"ticker","sector_i":"sector","centrality_i":"centrality"})
    side_j = df[["ticker_j","sector_j","centrality_j"]].rename(
        columns={"ticker_j":"ticker","sector_j":"sector","centrality_j":"centrality"})
    meta = pd.concat([side_i, side_j]).drop_duplicates("ticker")
    return meta.set_index("ticker")