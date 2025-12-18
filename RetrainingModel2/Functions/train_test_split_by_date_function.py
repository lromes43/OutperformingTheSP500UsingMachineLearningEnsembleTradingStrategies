def train_test_split_by_date_function(data_path, train_end_date, test_start_date):
    from datetime import date
    import pandas as pd
    
    """
    Split dataset into train and test sets based on explicit dates.

    Args:
        data_path (str): Path to feather file containing full dataset.
        train_end_date (str or pd.Timestamp or datetime.date): Last date for training set.
        test_start_date (str or pd.Timestamp or datetime.date): First date for test set.

    Returns:
        train_subset, test_subset (DataFrames)
    """
    data = pd.read_feather(data_path)
    
    # 1. Ensure the 'Date' column is a proper datetime object
    # The error message indicates it is already UTC-aware (e.g., datetime64[ns, UTC])
    data['Date'] = pd.to_datetime(data['Date']) 
    data = data.sort_values(by=['Ticker', 'Date'])

    # 2. Convert comparison dates to timezone-aware UTC objects (THE FIX)
    # The .tz_localize('UTC') method makes the naive date UTC-aware.
    train_end_date = pd.to_datetime(train_end_date).tz_localize('UTC')
    test_start_date = pd.to_datetime(test_start_date).tz_localize('UTC')

    # 3. Perform the comparison
    train_subset = data[data['Date'] <= train_end_date]
    test_subset = data[data['Date'] >= test_start_date]

    # Save to feather
    train_subset.to_feather("TrainData.feather")
    test_subset.to_feather("TestData.feather")

    return train_subset, test_subset





from datetime import date
data_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/CHECKFEATHER.feather"
train_end_date = date(2025,8,29)
test_start_date = date(2025,8,30)

train_subset, test_subset = train_test_split_by_date_function(
    data_path=data_path,
    train_end_date=train_end_date,
    test_start_date=test_start_date
)
