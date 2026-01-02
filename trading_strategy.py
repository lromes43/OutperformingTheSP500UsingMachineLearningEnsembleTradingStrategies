import sys
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Functions"))

from sp500_pipeline import run_sp500_pipeline
from split import train_test_split_by_date_function

def get_and_process_data(start_date, end_date):
    """Downloads data and calculates 325+ factors."""
    run_sp500_pipeline(start_date, end_date) 

def prepare_model_data(data_path, train_end, test_start):
    """Splits the processed feather file into train/test."""
    train_df, test_df = train_test_split_by_date_function(data_path, train_end, test_start)
    return train_df, test_df

if __name__ == "__main__":
    DATA_PATH = os.path.join(BASE_DIR, "Data", "FINALSP500Data.feather")
    get_and_process_data("2020-01-01", "2024-01-01")
    train, test = prepare_model_data(DATA_PATH, "2023-01-01", "2023-01-02")