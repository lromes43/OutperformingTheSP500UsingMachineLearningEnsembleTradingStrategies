import pytest
import pandas as pd
from unittest.mock import patch
from trading_strategy import get_and_process_data, prepare_model_data

@pytest.fixture
def mock_feather_data(tmp_path):
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2022-12-01', '2023-01-01', '2023-02-01']),
        'Ticker': ['AAPL', 'AAPL', 'AAPL'],
        'Close': [150, 155, 160]
    })
    file_path = tmp_path / "test_data.feather"
    df.to_feather(file_path)
    return str(file_path)

@patch('trading_strategy.run_sp500_pipeline')
def test_get_and_process_data_calls_pipeline(mock_pipeline):
    get_and_process_data("2023-01-01", "2023-01-05")
    mock_pipeline.assert_called_once_with("2023-01-01", "2023-01-05")

def test_prepare_model_data_logic(mock_feather_data):
    train_end = "2022-12-31"
    test_start = "2023-01-01"
    train, test = prepare_model_data(mock_feather_data, train_end, test_start)
    assert len(train) == 1 
    assert len(test) == 2   

def test_split_with_empty_date_range(mock_feather_data):
    train, test = prepare_model_data(mock_feather_data, "2025-01-01", "2025-01-02")
    assert len(test) == 0

