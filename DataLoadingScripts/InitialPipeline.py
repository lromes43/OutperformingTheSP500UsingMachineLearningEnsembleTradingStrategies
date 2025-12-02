import os
import time
import pandas as pd
import yfinance as yf


url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
sp500 = pd.read_csv(url)
tickers = sp500['Symbol'].tolist()

import yfinance as yf
import pandas as pd

def get_stock_data(tickers, start_date, end_date):
    all_data = []

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data for {ticker}")
            continue
        data['Ticker'] = ticker  # Add ticker column
        all_data.append(data.reset_index())

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Optional: reorder columns
    combined_df = combined_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    return combined_df


tickers = tickers
start_date = '2024-11-11'
end_date = '2025-11-10'

df = get_stock_data(tickers, start_date, end_date)
print(df.head())








test = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/Combined_SP500.csv")
test

#Train/ Test Split

from datetime import date

import pandas as pd

import pandas as pd

import pandas as pd

def train_test_split_by_date(data_path, train_end_date, test_start_date):
    """
    Split dataset into train and test sets based on explicit dates.

    Args:
        data_path (str): Path to feather file containing full dataset.
        train_end_date (str or pd.Timestamp): Last date for training set.
        test_start_date (str or pd.Timestamp): First date for test set.

    Returns:
        train_subset, test_subset (DataFrames)
    """
    data = pd.read_feather(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by=['Ticker', 'Date'])

    train_end_date = pd.to_datetime(train_end_date)
    test_start_date = pd.to_datetime(test_start_date)

    train_subset = data[data['Date'] <= train_end_date]
    test_subset = data[data['Date'] >= test_start_date]

    # Save to feather
    train_subset.to_feather("TrainData.feather")
    test_subset.to_feather("TestData.feather")

    return train_subset, test_subset


data_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/FinalTestData.feather"
train_end_date = "2025-08-29"
test_start_date = "2025-08-30"

train_subset, test_subset = train_test_split_by_date(
    data_path=data_path,
    train_end_date=train_end_date,
    test_start_date=test_start_date
)

pd.read_feather('/Users/lukeromes/Desktop/Personal/Sp500Project/TestData.feather')
pd.read_feather('/Users/lukeromes/Desktop/Personal/Sp500Project/TrainData.feather')






test_train = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/TrainData.feather")
test_test = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/TestData.feather")









data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/FinalTestData.feather")
data