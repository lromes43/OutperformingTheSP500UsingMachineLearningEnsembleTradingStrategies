def train_test_split_by_date_function(data_path, train_end_date, test_start_date):
    import pandas as pd
    
    data = pd.read_feather(data_path)
    data['Date'] = pd.to_datetime(data['Date'])

    t_end = pd.to_datetime(train_end_date).tz_localize(None)
    v_start = pd.to_datetime(test_start_date).tz_localize(None)

    data = data.sort_values(by=['Ticker', 'Date'])

    train_subset = data[data['Date'] <= t_end].copy()
    test_subset = data[data['Date'] >= v_start].copy()

    base_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/"
    train_subset.to_feather(f"{base_path}TrainData.feather")
    test_subset.to_feather(f"{base_path}TestData.feather")

    print(f"Split complete. Train shape: {train_subset.shape}, Test shape: {test_subset.shape}")
    return train_subset, test_subset