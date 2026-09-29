import numpy as np
import pandas as pd
import yfinance as yf
from market_ml.config import * 
import os
from IPython.display import display 

class FeatureEngineer:

    def __init__(self, data: pd.DataFrame, benchmark_data:pd.DataFrame | None=None):
        self.data = data.copy()

        if self.benchmark_data is not None:
            self.benchmark_data = benchmark_data.copy()

    def add_daily_returns(self):
        '''
        Calculates and adds daily returns.
        '''

        self.data['Daily_Return'] = self.data.groupby("Ticker")["Close"].pct_change()
        return self.data


    def add_return_feature(self, period=5):
        '''
        Calculates and adds returns for a given period.
        '''

        self.data[f"Return_{period}D"] = self.data.groupby("Ticker")["Close"].pct_change(periods=period)
        return self.data

    def add_volatility_feature(self, w=20):
        '''
        Calculates and adds returns for a given period.
        '''
    
        self.data[f"Volatility_{w}D"] = self.data.groupby("Ticker")['Daily_Return'].rolling(w).std().reset_index(level=0, drop=True)
        return self.data

    def add_SMA_feature(self, w=20):
        '''
        Calculates the moving average for a given window size
        '''
        self.data[f"SMA_{w}D"] = self.data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=w).mean())
        return self.data

    def add_price_relative_to_sma(self,w=20):
        '''
        Calculating price relative to its moving average for a certain window size
        '''
        self.data[f"Price_to_SMA_{w}D"] = self.data["Close"]/self.data[f"SMA_{w}D"]
        return self.data

    def add_average_volume_feature(self, w=20):
        '''
        Calculates the volume moving for a given period.
        '''
        self.data[f"Volume_Average_{w}D"] = self.data.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(window=w).mean())
        return self.data
    
    def add_relative_volume(self, w=20):
        '''
        Calculates volume relative to the average for a given period. 
        '''
        self.data[f"Relative_Volume_{w}D"] = self.data["Volume"]/self.data[f"Volume_Average_{w}D"]
        return self.data

    