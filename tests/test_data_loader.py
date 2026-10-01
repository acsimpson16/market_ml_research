from market_ml.data_loader import DataLoader


test_tickers = ["AAPL", "NVDA"]
benchmark_tickers = ["SPY"]
test_period = "10y"
test_interval = "1d"

def test_download_data():
    dl = DataLoader(tickers=test_tickers, period=test_period, interval=test_interval, file_name='test.parquet')
    benchmark_dl = DataLoader(tickers=benchmark_tickers, period=test_period, interval=test_interval, file_name='benchmark_test.parquet')
    data = dl.download_data()
    benchmark_data = benchmark_dl.download_data()
    assert not data.empty, "Downloaded data should not be empty"
    assert not benchmark_data.empty, "Downloaded benchmark data should not be empty"
    assert "Close" in data.columns.get_level_values(0), "Downloaded data should contain 'Close' column"
    assert "Close" in benchmark_data.columns.get_level_values(0), "Downloaded benchmark data should contain 'Close' column"

def test_process_data():
    dl = DataLoader(tickers=test_tickers, period=test_period, interval=test_interval, file_name='test.partquet')
    raw_data = dl.download_data()
    processed_data = dl.process_data(raw_data)

    assert not processed_data.empty, "Processed data should not be empty"
    assert "Ticker" in processed_data.columns, "Processed data should contain 'Ticker' column"
    assert "Sector" in processed_data.columns, "Processed data should contain 'Sector' column"

def test_check_data():

    dl = DataLoader(tickers=test_tickers, period=test_period, interval=test_interval, file_name='test.parquet')
    raw_data = dl.download_data()
    processed_data = dl.process_data(raw_data)
    
    validation = dl.check_data(processed_data)

    assert validation.loc[test_tickers[0], "Missing_Close"] == 0
    assert "Missing_Close"in validation.columns, "validation data should contain 'Missing_Close' column"