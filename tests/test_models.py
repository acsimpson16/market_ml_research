from market_ml.data_loader import DataLoader
from market_ml.EDA import * 
from market_ml.models import *

def test_split_chronological_data():
    dates = pd.date_range(start="2025-01-01", periods=10, freq="D")
    data_size = len(dates)
    
    test_df = pd.DataFrame({
    "Date": dates,
    "Ticker": ["AAPL"] * data_size,
    "Close": range(100, 100 + data_size)
    })

    test_split = 0.8

    x, y, split_date = split_chronological_data(test_df, test_split, gap_days=0)

    #can't have empty dataframes    
    assert not test_df.empty, "Dataframe can not be empty"
    assert not x.empty, "training data cannot be empty"
    assert not y.empty, "testing data cannot be empty"

    #split data should be chronological
    assert x["Date"].min() < x["Date"].max(), "dataframe must be chronological"
    assert y["Date"].min() > x["Date"].max(), "Testing data must start after the trainging data"

    # 10 days should be split into 8 days for training and 2 days for testing
    assert len(x) == 8
    assert len(y) == 2


def test_split_chronological_data_with_gap():
    dates = pd.date_range(start="2025-01-01", periods=20, freq="D")
    data_size = len(dates)

    test_df = pd.DataFrame({
        "Date": dates,
        "Ticker": ["AAPL"] * data_size,
        "Close": range(100, 100 + data_size)
    })

    train_data, test_data, split_date = split_chronological_data(
        test_df,
        split_ratio=0.8,
        gap_days=5
    )

    # Training and testing data should not be empty
    assert not train_data.empty
    assert not test_data.empty

    # Training data must end before the testing period
    assert train_data["Date"].max() < test_data["Date"].min()

    # There should be exactly five dates excluded between them
    excluded_dates = pd.date_range(
        start=train_data["Date"].max() + pd.Timedelta(days=1),
        end=test_data["Date"].min() - pd.Timedelta(days=1)
    )

    assert len(excluded_dates) == 5

    # The returned split date should match the first testing date
    assert split_date == test_data["Date"].min()
