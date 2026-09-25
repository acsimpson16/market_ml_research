import numpy as np
import pandas as pd
import yfinance as yf
import os



class DataLoader:
    def __init__(self, ticker=None, period=None, interval=None, data_path=None):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = None
        self.data_path = "data/raw"
        

    def download_data(self):
        data = yf.download(self.ticker, period=self.period, interval=self.interval)
        data.columns = data.columns.get_level_values(0) #data has multi-index, second is the ticker which is being removed here. 
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
        validation = {'Missing Values': self.data.isna().sum().sum(),
                  'Number of Rows': len(self.data),
                  'Number of Columns': len(self.data.columns),
                  'Start': pd.to_datetime(self.data.min()),
                  'End': pd.to_datetime(self.data.index.max()),
                  'Duplicate Dates': self.data.index.duplicated().sum()
                  
                  
    }   
        #validation_df = pd.DataFrame(index=validation.keys(), validation)
        return validation