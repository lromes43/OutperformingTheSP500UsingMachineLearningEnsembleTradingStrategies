def train_test_split_by_date_function(data_path, train_end_date, test_start_date):
    from datetime import date
    import pandas as pd
    data = pd.read_feather(data_path)
    
    data['Date'] = pd.to_datetime(data['Date']) 
    data = data.sort_values(by=['Ticker', 'Date'])

    train_end_date = pd.to_datetime(train_end_date).tz_localize('UTC')
    test_start_date = pd.to_datetime(test_start_date).tz_localize('UTC')

    train_subset = data[data['Date'] <= train_end_date]
    test_subset = data[data['Date'] >= test_start_date]

    train_subset.to_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2/Data/TrainData.feather")
    test_subset.to_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2/Data/TestData.feather")

    return train_subset, test_subset





