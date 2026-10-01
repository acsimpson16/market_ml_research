from market_ml.data_loader import DataLoader
from market_ml.EDA import * 


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





def test_calculate_returns():
    data = stock_data
    returns = calculate_returns(data)
    assert not data.empty, "Downloaded data should not be empty"
    assert "Daily_Return" in returns.columns, "Daily Returns data should contain 'Daily_Return' column"

def test_calculate_volatility():
    data = stock_data
    returns = calculate_returns(data)
    volatility = calculate_rolling_volatility(returns, window=20)
    assert not returns.empty, "Returns data should not be empty"
    assert "Daily_Return" in volatility.columns, "Volatility data should contain 'Daily_Return' column"
    assert "Volatility_20D" in volatility.columns, "Volatility data should contain 'Volatility' column"


def test_calculate_drawdown():
    data = stock_data
    returns = calculate_returns(data)
    drawdown = calculate_drawdown(returns)
    assert not returns.empty, "Returns data should not be empty"
    assert "Daily_Return" in drawdown.columns, "drawdown dataframe should contain 'Daily_Return' column" 
    assert "Drawdown" in drawdown.columns, "drawdown dataframe should contain 'Drawdown'column"
    assert (drawdown["Drawdown"].dropna() <= 0.0).all(), 'Drawdown values should not be positive' 


