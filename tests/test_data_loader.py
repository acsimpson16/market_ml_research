from market_ml.data_loader import DataLoader

def test_download_data():
    dl = DataLoader(ticker="SPY", start_date="2020-01-01")
    data = dl.download_data()
    assert not data.empty, "Downloaded data should not be empty"
    assert "Close" in data.columns, "Downloaded data should contain 'Close' column"