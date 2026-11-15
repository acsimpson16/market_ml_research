from market_ml.data_loader import DataLoader
from market_ml.EDA import * 
from market_ml.models import *


def test_split_chronological_data():
    dates = pd.date_range(start="2025-01-01", periods=20, freq="D")

    test_df = pd.DataFrame({
        "Date": dates,
        "Ticker": ["AAPL"] * len(dates),
        "Close": range(100, 100 + len(dates))
    })

    train_data, validation_data, test_data = split_chronological_data(
        test_df,
        train_ratio=0.70,
        validation_ratio=0.15,
        gap_days=0
    )

    # No dataset should be empty
    assert not train_data.empty
    assert not validation_data.empty
    assert not test_data.empty

    # All datasets should be chronological
    assert train_data["Date"].is_monotonic_increasing
    assert validation_data["Date"].is_monotonic_increasing
    assert test_data["Date"].is_monotonic_increasing

    # Datasets must not overlap
    assert train_data["Date"].max() < validation_data["Date"].min()
    assert validation_data["Date"].max() < test_data["Date"].min()

    # 70% / 15% / 15% split of 20 dates
    assert len(train_data) == 14
    assert len(validation_data) == 3
    assert len(test_data) == 3


def test_split_chronological_data_with_gap():
    dates = pd.date_range(start="2025-01-01", periods=40, freq="D")

    test_df = pd.DataFrame({
        "Date": dates,
        "Ticker": ["AAPL"] * len(dates),
        "Close": range(100, 100 + len(dates))
    })

    train_data, validation_data, test_data = split_chronological_data(
        test_df,
        train_ratio=0.70,
        validation_ratio=0.15,
        gap_days=5
    )

    # Training, validation and testing data should not be empty
    assert not train_data.empty
    assert not validation_data.empty
    assert not test_data.empty

    # Training must end before validation; validation before testing
    assert train_data["Date"].max() < validation_data["Date"].min()
    assert validation_data["Date"].max() < test_data["Date"].min()

    # Exactly five dates should be excluded between each dataset
    train_validation_gap = pd.date_range(
        start=train_data["Date"].max() + pd.Timedelta(days=1),
        end=validation_data["Date"].min() - pd.Timedelta(days=1)
    )

    validation_test_gap = pd.date_range(
        start=validation_data["Date"].max() + pd.Timedelta(days=1),
        end=test_data["Date"].min() - pd.Timedelta(days=1)
    )

    assert len(train_validation_gap) == 5
    assert len(validation_test_gap) == 5
