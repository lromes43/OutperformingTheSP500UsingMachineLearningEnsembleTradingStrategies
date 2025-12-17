import pandas as pd
import yfinance as yf
import numpy as np
import os
import time
import random
import re
from datetime import datetime

def run_sp500_pipeline(start_date, end_date):
    base_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2"
    download_dir = os.path.join(base_path, "Data/Pulling")
    combined_csv = os.path.join(base_path, "Data/combined_sp500_data.csv")
    final_feather_file = os.path.join(base_path, "FinalTestData.feather")
    
    os.makedirs(download_dir, exist_ok=True)
    
    print("Fetching S&P 500 ticker list...")
    try:
        sp500_url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
        sp500 = pd.read_csv(sp500_url)
        tickers = sp500['Symbol'].tolist()
    except Exception as e:
        print(f"Error loading S&P 500 list: {e}")
        return

    print(f"Starting download for {len(tickers)} companies from {start_date} to {end_date}...")
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            sleep_time = random.uniform(20, 30)
            print(f"\n☕ API Break: Resting for {sleep_time:.1f}s...")
            time.sleep(sleep_time)

        if ticker == 'IPG': continue
        
        file_path = os.path.join(download_dir, f"{ticker}.csv")
        
        for attempt in range(3):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
                if not data.empty:
                    data['Ticker'] = ticker
                    data.to_csv(file_path)
                    break
            except Exception:
                time.sleep(2)
        
        print(f"Progress: {i+1}/{len(tickers)}", end='\r')
        time.sleep(random.uniform(0.5, 1.5))

    print("\nMerging files...")
    all_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith('.csv')]
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values(['Ticker', 'Date'])

    print("Calculating technical indicators...")
    
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    df['Day_Range'] = df['High'] - df['Low']
    
    for window in [5, 12, 15, 20, 26, 30, 50]:
        df[f'SMA_{window}'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    for window in [5, 12, 26, 50]:
        df[f'EMA_{window}'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=window, adjust=False).mean())

    df['MACD'] = df['EMA_12'] - df['EMA_26']

    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = df.groupby('Ticker')['Close'].transform(calc_rsi)

    df['SD'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=20).std())
    df['Bollinger_Band_Middle'] = df['SMA_20']
    df['Bollinger_Band_Upper'] = df['SMA_20'] + (df['SD'] * 2)
    df['Bollinger_Band_Lower'] = df['SMA_20'] - (df['SD'] * 2)

    def calc_obv(group):
        return (np.sign(group['Close'].diff()).fillna(0) * group['Volume']).cumsum()
    df['OBV'] = df.groupby('Ticker', group_keys=False).apply(calc_obv)


    print("Adding VIX data...")
    vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d", progress=False)['Close']
    vix = vix.reset_index()
    vix.columns = ['Date', 'VIX_Close']
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
    df = pd.merge(df, vix, on='Date', how='left')

    print("Calculating target labels...")
    df['next_day_pct_change'] = df.groupby('Ticker')['Close'].shift(-1).pct_change() 
    df['next_day_pct_change'] = (df.groupby('Ticker')['Close'].shift(-1) - df['Close']) / df['Close']
    df['Movement'] = (df['next_day_pct_change'] > 0).astype(int)
    
    df['next_5_day_pct_change'] = (df.groupby('Ticker')['Close'].shift(-5) - df['Close']) / df['Close']
    df['Movement_5_day'] = (df['next_5_day_pct_change'] > 0).astype(int)


    print(f"Saving to {final_feather_file}...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype('float32')
    
    df.reset_index(drop=True).to_feather(final_feather_file)
    print("Pipeline Complete!")


if __name__ == "__main__":
    d1 = "2024-01-01" 
    d2 = "2024-12-01"
    run_sp500_pipeline(start_date=d1, end_date=d2)