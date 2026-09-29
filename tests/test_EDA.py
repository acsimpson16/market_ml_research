from market_ml.data_loader import DataLoader
from market_ml.EDA import * 


test_tickers = ["AAPL", "NVDA"]
test_period = "10y"
test_interval = "1d"



def test_calculate_returns():
    dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = dl.load_data()
    returns = calculate_returns(data)
    assert not data.empty, "Downloaded data should not be empty"
    assert "Daily_Return" in returns.columns, "Daily Returns data should contain 'Daily_Return' column"

def test_calculate_volatility():
    dl = DataLoader(tickers=test_tickers, period="10y", interval="1d")
    data = dl.load_data()
    returns = calculate_returns(data)
    volatility = calculate_rolling_volatility(returns, window=20)
    assert not returns.empty, "Returns data should not be empty"
    assert "Daily_Return" in volatility.columns, "Volatility data should contain 'Daily_Return' column"
    assert "Volatility_20D" in volatility.columns, "Volatility data should contain 'Volatility' column"

