import pandas as pd
import os

def train_test_split_by_date_function(data_path, train_end_date, test_start_date):
    data = pd.read_feather(data_path)
    data['Date'] = pd.to_datetime(data['Date'])

    t_end = pd.to_datetime(train_end_date).tz_localize(None)
    v_start = pd.to_datetime(test_start_date).tz_localize(None)

    data = data.sort_values(by=['Ticker', 'Date'])

    train_subset = data[data['Date'] <= t_end].copy()
    test_subset = data[data['Date'] >= v_start].copy()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "Data")


    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_subset.to_feather(os.path.join(data_dir, "TrainData.feather"))
    test_subset.to_feather(os.path.join(data_dir, "TestData.feather"))

    print(f"Split complete. Train shape: {train_subset.shape}, Test shape: {test_subset.shape}")
    return train_subset, test_subset