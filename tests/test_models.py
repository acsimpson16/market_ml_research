from market_ml.data_loader import DataLoader
from market_ml.EDA import * 
from market_ml.models import *

def test_split_chronological_data():
    dates = pd.date_range(start="2025-01-01",end="2025-01-10")
    data_size = len(dates)
    ticker = ['AAPL']*data_size
    close = [100, 400, 200, 100, 300]*2

    test_split = 0.8

    test_df = pd.DataFrame({"Date":dates, "Ticker":ticker, "Close":close})

    x, y = split_chronological_data(test_df, test_split)

    
    assert not test_df.empty, "Dataframe can not be empty"
    assert not x.empty, "training data cannot be empty"
    assert not y.empty, "testing data cannot be empty"

    assert x["Date"].min() < x["Date"].max(), "dataframe must be chronological"
    assert y["Date"].min() > x["Date"].max(), "Testing data must start after the trainging data"

    assert len(x) == 8
    assert len(y) == 2