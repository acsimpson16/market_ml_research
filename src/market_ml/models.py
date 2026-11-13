import numpy as np
import pandas as pd
import yfinance as yf
from market_ml.config import * 
import os
from IPython.display import display
from sklearn.model_selection import TimeSeriesSplit 

def split_chronological_data(data, split_ratio=0.8, gap_days=5):
    '''
    Takes a dataframe and splits the data into training and test data for a given split ratio
    '''
    dates = sorted(data["Date"].dropna().unique())
    split_idx = int(len(dates) * split_ratio)

    test_start_date = dates[split_idx]
    train_end_date = dates[split_idx - gap_days]

    train_data = data[data["Date"] < train_end_date].copy()
    test_data = data[data["Date"] >= test_start_date].copy()
    
    return train_data, test_data, test_start_date

def date_based_splits(data, n_splits=5):
    '''
    Creates a time series CV by date rather than position.
    '''
    dates = np.array(sorted(data["Date"].unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, val_idx in splitter.split(dates):
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]

        train_rows = np.flatnonzero(
            data["Date"].isin(train_dates).to_numpy()
        )
        val_rows = np.flatnonzero(
            data["Date"].isin(val_dates).to_numpy()
        )

        yield train_rows, val_rows #generates a sequence of results at one time and remembers it's state