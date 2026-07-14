import numpy as np
import pandas as pd
import yfinance as yf
import os

class DataLoader:
    def __init__(self, ticker=None, start_date=None, end_date=None, data_path=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
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
        
    def load_data(self):
        if os.path.exists(self.data_path + f"{self.ticker}_data.csv"):
            self.data = pd.read_csv(self.data_path + f"{self.ticker}_data.csv", index_col=0, parse_dates=True)
            print(f"Data loaded for {self.ticker}")
            return self.data
        else:
            raise FileNotFoundError(f"No data found for {self.ticker}. Please download the data first.")
    
    def check_data(self):
        validation = {'Number of NaNs': self.data.isna().sum().sum(),
                  'Number of Rows': len(self.data),
                  'Number of Columns': len(self.data.columns),
                  'Start': pd.to_datetime(self.data.index.min()),
                  'End': pd.to_datetime(self.data.index.max()),
                  
                  
    }   
        #validation_df = pd.DataFrame(index=validation.keys(), validation)
        return validation