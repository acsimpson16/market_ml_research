from market_ml.data_loader import DataLoader
from market_ml.EDA import * 
from market_ml.feature_engineering import *


test_tickers = ["AAPL", "NVDA"]
test_period = "10y"
test_interval = "1d"

stock_data = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01","2025-01-02","2025-01-03",]),
                           "Ticker": ["AAPL", "AAPL", "AAPL"],
                           "Close": [100, 105, 110], 
                           "Volume":[100, 200, 300]
                            })
benchmark_data = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01","2025-01-02","2025-01-03",] * 2),
                               "Ticker": ["SPY"] * 3 + ["QQQ"] * 3,
                               "Close": [100, 110, 121, 200, 220, 242], "Volume":[200, 400, 200, 600, 300, 250]
                               })


def test_daily_returns_feature():
    #dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = stock_data
    
    fe = FeatureEngineer(data)
    daily_returns = fe.add_daily_returns()
    assert not daily_returns.empty, "returns should not be empty"
    assert f"Daily_Return" in daily_returns.columns, "Returns Feature data should have 'Daily_Return' column"


def test_add_returns_feature():
    #dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = stock_data
    p = 5
    fe = FeatureEngineer(data)
    returns = fe.add_return_feature(period=p)
    assert not returns.empty, "returns should not be empty"
    assert f"Return_{p}D" in returns.columns, "Returns Feature data with period of 5 should contain 'Returns' column"

def test_add_volatility_feature():
    #dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = stock_data
    w = 20
    fe = FeatureEngineer(data)
    returns = fe.add_daily_returns()
    volatility = fe.add_volatility_feature(w)
    assert not volatility.empty, "volatility should not be empty"
    assert f"Volatility_{w}D" in volatility.columns, "Volatility Feature data with window of 20 should contain 'Volatility' column"


def test_add_sma_feature():
    #dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = stock_data
    w = 20
    fe = FeatureEngineer(data)
    sma = fe.add_SMA_feature(20)

    assert not sma.empty, "SMA should not be empty"
    assert "SMA_20D" in sma.columns, "SMA data should have 'SMA20D' column"

def test_add_sma_feature():
    #dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = stock_data
    w = 20
    fe = FeatureEngineer(data)
    sma = fe.add_SMA_feature(w)
    price_sma = fe.add_price_relative_to_sma(w)


    assert not price_sma.empty, "Price_SMA should not be empty"
    assert "Price_to_SMA_20D" in sma.columns, "Price_SMA data should have 'Price_to_SMA_20D' column"

def test_average_volume():
    #dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = stock_data
    w = 20
    fe = FeatureEngineer(data)
    volume = stock_data["Volume"]
    average_volume = fe.add_average_volume_feature(w)
    assert not average_volume.empty, "average volume should not be empty"
    assert "Volume_Average_20D" in average_volume.columns, "Average volume should have 'Volume_Average_20D'"
    return

def test_relative_volume():
    #dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = stock_data
    w = 20
    fe = FeatureEngineer(data)
    average_volume = fe.add_average_volume_feature(w)
    relative_volume = fe.add_relative_volume(w)
    assert not relative_volume.empty, "Relative volume should not be empty"
    assert "Relative_Volume_20D" in relative_volume.columns, "Relative volume should have 'Relative_Volume' column"

def test_benchmark_returns():
    sdata = stock_data
    bdata = benchmark_data

    fe = FeatureEngineer(sdata, bdata)
    benchmark_returns = fe.add_benchmark_returns()

    assert "SPY_Return" in benchmark_returns.columns
    assert "QQQ_Return" in benchmark_returns.columns