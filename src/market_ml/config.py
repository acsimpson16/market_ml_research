import numpy as np
import pandas as pd
import yfinance as yf
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TICKER_SECTOR = {# Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "META": "Technology",
    "GOOGL": "Technology",

    # Financials
    "JPM": "Financials",
    "GS": "Financials",
    "BAC": "Financials",
    "MS": "Financials",
    "BLK": "Financials",

    # Healthcare
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "ABBV": "Healthcare",
    "PFE": "Healthcare",
    "MRK": "Healthcare",

    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",

    # Consumer
    "AMZN": "Consumer",
    "COST": "Consumer",
    "WMT": "Consumer",
    "HD": "Consumer",
    "MCD": "Consumer",}


TICKERS = list(TICKER_SECTOR.keys())

BENCHMARK_TICKERS = ["SPY", "QQQ"]