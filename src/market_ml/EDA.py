import numpy as np
import pandas as pd
import yfinance as yf
from market_ml.config import * 
import os
from IPython.display import display 

def calculate_returns(data):
    '''
    Calculates daily returns for each ticker
    '''
    data['Daily_Return'] = data.groupby("Ticker")["Close"].pct_change()
    return data

def calculate_rolling_volatility(data, window=20):
    '''
    Calculates rolling volatility for each ticker for a given window size
    '''

    data[f"Volatility_{window}D"] = data.groupby("Ticker")["Daily_Return"].rolling(window=20).std().reset_index(level=0, drop=True)
    return data

def calculate_drawdown(data):
    '''
    Calculates drawdown for each ticker
    '''
    data["Cumulative_Returns"] = data.groupby("Ticker")["Daily_Return"].transform(lambda x: (1+x).cumprod())

    data["Highest_Peak"] = data.groupby("Ticker")["Cumulative_Returns"].cummax()

    data["Drawdown"] = (data["Cumulative_Returns"]-data["Highest_Peak"])/data["Highest_Peak"]
    return data