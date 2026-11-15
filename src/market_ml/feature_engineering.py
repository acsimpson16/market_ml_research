import numpy as np
import pandas as pd
import yfinance as yf
from market_ml.config import * 
import os
from IPython.display import display 

class FeatureEngineer:

    def __init__(
        self,
        data: pd.DataFrame,
        benchmark_data: pd.DataFrame | None = None,
        filename: str | None=None
    ):
        self.data = data.copy()
        self.data = self.data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        self.tickers = self.data
        if benchmark_data is not None:
            self.benchmark_data = benchmark_data.copy()
        else:
            self.benchmark_data = None

        self.filename = filename


    
    def add_daily_returns(self):
        '''
        Calculate daily returns separately for each ticker.
        '''

        self.data["Daily_Return"] = (
        self.data.groupby("Ticker")["Close"].pct_change())
        return self.data


    def add_return_feature(self, period=5):
        '''
        Calculate trailing returns over a given number of trading sessions.
        '''
        self.data[f"Return_{period}D"] = (
        self.data.groupby("Ticker")["Close"].pct_change(periods=period))
        return self.data

    def add_volatility_feature(self, w=20):
        '''
        Calculate rolling standard deviation of daily returns.
        '''
        self.data[f"Volatility_{w}D"] = (self.data.groupby("Ticker")["Daily_Return"].transform(lambda x: x.rolling(window=w).std()))
        return self.data


    def add_SMA_feature(self, w=20):
        '''
        Calculate a trailing simple moving average of closing prices.
        '''
        self.data[f"SMA_{w}D"] = (self.data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=w).mean()))
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


    def add_benchmark_returns(self):
        '''
        Calculates daily returns for benchmark indices and merge them into the stock dataframe.
        '''

        if self.benchmark_data is None:
            raise ValueError("Benchmark data is required.")

        benchmark_tickers = self.benchmark_data["Ticker"].unique()

        for ticker in benchmark_tickers:

            benchmark = self.benchmark_data[
                self.benchmark_data["Ticker"] == ticker
            ].copy()

            benchmark[f"{ticker}_Return"] = (
                benchmark["Close"].pct_change()
            )

            self.data = self.data.merge(
                benchmark[["Date", f"{ticker}_Return"]],
                on="Date",
                how="left"
            )


        print("Rows:", len(self.data))
        print("Duplicate ticker/date pairs:", self.data.duplicated(["Ticker", "Date"]).sum())
        return self.data

    def add_target_variable(self, horizon=5):

        '''
        Calculates the returns and if the returns are positive over a given horizon
        '''

        #calculating the returns over a given horizon
        self.data[f"Future_Return_{horizon}D"] = (self.data.groupby("Ticker", sort=False)["Close"].shift(-horizon) / self.data["Close"] - 1)

        #calculating if the returns are positive (1) or negative (0) over a given horizon
        self.data[f"Target_{horizon}D_Direction"] = self.data[f"Future_Return_{horizon}D"].gt(0).where(self.data[f"Future_Return_{horizon}D"].notna()).astype("Int64")
        return self.data 

    def save_features(self):
        self.data.to_parquet(PROCESSED_DATA_DIR / self.filename)
        print(f"Features data saved for {self.tickers}.")
        return

    def load_features(self):
        if os.path.exists(PROCESSED_DATA_DIR/self.filename):
                    self.data = pd.read_parquet(PROCESSED_DATA_DIR/self.filename)
                    print(f"Data loaded for {self.tickers}.")
                    return self.data
        else:
            raise FileNotFoundError(f"No data found for {self.tickers}. Please download the data first.")