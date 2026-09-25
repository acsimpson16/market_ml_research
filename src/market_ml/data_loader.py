import numpy as np
import pandas as pd
import yfinance as yf
from market_ml.config import * 
import os
from IPython.display import display 




class DataLoader:
    def __init__(self, 
                 tickers: list[str] | None =None, 
                 period: str | None=None, 
                 interval: str | None=None
                 ):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.data = None
        

    def download_data(self) -> pd.DataFrame:
        '''
        Downloads, processes, and optionally saves market data for a given list of tickers 
        over a specified period and interval.
        '''
        
        raw_data = yf.download(self.tickers, period=self.period, interval=self.interval)
        raw_data.to_parquet(RAW_DATA_DIR / "market_data.parquet")
        print(f"Downloaded data for {self.tickers}")

        return raw_data
    
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame: 
        processed_data = raw_data.stack(level="Ticker", future_stack=True).reset_index()
        processed_data["Sector"] = processed_data["Ticker"].map(TICKER_SECTOR)
        processed_data = processed_data[['Date', 'Ticker', 'Sector', 'Close', 'High', 'Low', 'Open', 'Volume']]
        self.data = processed_data
        print(f'Processed data for {self.tickers}')
        #if self.save_data:
        #    processed_data.to_parquet(PROCESSED_DATA_DIR / "market_data.parquet")
        #    print(f'Saved data for {self.tickers}')

        return self.data

    def check_data(self, data: pd.DataFrame, ticker_summary: bool = False) ->  dict | tuple[dict, pd.DataFrame]:
        
        validation = {
            "Missing Values": data.isna().sum().sum(),
            "Number of Rows": len(data),
            "Number of Columns": len(data.columns),
            "Start": data.index.min(),
            "End": data.index.max(),
            "Duplicate Rows": data.duplicated().sum(),
            "Duplicate Date/Ticker": data.duplicated(subset=["Date", "Ticker"]).sum(),
        }
        print(' '*40)
        print('DATA VALIDATION:')
        
        for key, value in validation.items():
            print(f"{key}: {value}")
        #display(validation)

        if ticker_summary:
            ticker_validation = (
                data.groupby("Ticker")
                .agg(
                    Rows=("Ticker", "size"),
                    Start=("Date", "min"),
                    End=("Date", "max"),
                    Missing_Close=("Close", lambda x: x.isna().sum())
                )
            )
            print(' '*40)
            print('ticker_validation')
            display(ticker_validation)
            return 

        return 

    def save_data(self):
        self.data.to_parquet(PROCESSED_DATA_DIR / "market_data.parquet")
        print(f'Process data saved for {self.tickers}')
        return
    
    def load_data(self) -> pd.DataFrame:
        """
        Load processed market data from Parquet.

        """
        if os.path.exists(PROCESSED_DATA_DIR/"market_data.parquet"):
            self.data = pd.read_parquet(PROCESSED_DATA_DIR/"market_data.parquet")
            print(f"Data loaded for {self.tickers}")
            return self.data
        else:
            raise FileNotFoundError(f"No data found for {self.tickers}. Please download the data first.")
    
