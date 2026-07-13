import numpy as np
import pandas as pd
import yfinance as yf
import os

class DataLoader:
    def __init__(self, ticker=None, start_date=None, end_date=None, data_path=None):
        self.ticker = "SPY"
        self.start_date = "2010-01-01"
        self.end_date = None
        self.data = None
        self.data_path = "data/raw"
        

    def download_data(self):
        #print(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}")
        stock = yf.Ticker(self.ticker)
        data = stock.history(start=self.start_date, end=self.end_date)
        self.data = data
        print(f"Downloaded data for {self.ticker}")
        return self.data
    
    def save_data(self):
        if self.data is not None:
            os.makedirs(self.data_path, exist_ok=True)
            self.data.to_csv(self.data_path + f"{self.ticker}_data.csv")
            print(f"Data saved for {self.ticker}")
        else:
            raise RuntimeError("Download data before saving.")

    