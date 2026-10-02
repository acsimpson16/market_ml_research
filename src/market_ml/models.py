import numpy as np
import pandas as pd
import yfinance as yf
from market_ml.config import * 
import os
from IPython.display import display 

def split_chronological_data(data, split_ratio=0.8):
    '''
    Takes a dataframe and splits the data into training and test data for a given split ratio
    '''
    split_idx = int(split_ratio * len(data))
    split_date = list(data["Date"])[split_idx]
    train_data = data[data["Date"] < split_date].copy()
    test_data = data[data["Date"] >= split_date].copy()
    return train_data, test_data, split_date