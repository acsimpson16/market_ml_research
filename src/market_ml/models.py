import numpy as np
import pandas as pd
import yfinance as yf
from market_ml.config import * 
import os
from IPython.display import display
from sklearn.model_selection import TimeSeriesSplit 

def split_chronological_data(data, train_ratio=0.70, validation_ratio=0.15, gap_days=5):
    """
    Splits data chronologically into train, validation and test
    """

    dates = sorted(data["Date"].dropna().unique())

    train_idx = int(len(dates) * train_ratio)
    validation_idx = int(len(dates) * (train_ratio + validation_ratio))

    train_end_date = dates[train_idx - gap_days - 1]
    validation_start_date = dates[train_idx]
    validation_end_date = dates[validation_idx - gap_days - 1]
    test_start_date = dates[validation_idx]

    train_data = data[data["Date"] <= train_end_date].copy()

    validation_data = data[
        (data["Date"] >= validation_start_date) &
        (data["Date"] <= validation_end_date)
    ].copy()

    test_data = data[data["Date"] >= test_start_date].copy()

    return train_data, validation_data, test_data

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