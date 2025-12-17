import pandas as pd
import yfinance as yf
import numpy as np
import os
import time
import random
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_sp500_pipeline(start_date, end_date):
    base_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2"
    download_dir = os.path.join(base_path, "Data/Pulling")
    final_feather_file = os.path.join(base_path, "FinalTestData.feather")
    
    os.makedirs(download_dir, exist_ok=True)
    
    print("Fetching S&P 500 ticker list...")
    sp500_url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
    sp500 = pd.read_csv(sp500_url)
    tickers = sp500['Symbol'].tolist()

    print(f"Downloading data for {len(tickers)} companies...")
    for i, ticker in enumerate(tickers):
        if ticker in ['IPG', 'BRK.B', 'BF.B']: continue
        file_path = os.path.join(download_dir, f"{ticker}.csv")
        
        if (i + 1) % 50 == 0: 
            time.sleep(random.uniform(20, 30))
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data['Ticker'] = ticker
                data.to_csv(file_path)
        except Exception: 
            pass
        print(f"Progress: {i+1}/{len(tickers)} - {ticker}", end='\r')

    print("\nMerging and fixing data types...")
    all_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith('.csv')]
    raw_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in temp_df.columns:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        raw_list.append(temp_df)
    
    df = pd.concat(raw_list, ignore_index=True)
    df = df.dropna(subset=['Close'])
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values(['Ticker', 'Date'])

    print("Calculating technical indicators...")
    group = df.groupby('Ticker')
    
    df['Daily_Return'] = group['Close'].pct_change() * 100
    df['Day_Range'] = df['High'] - df['Low']
    
    for w in [5, 12, 15, 20, 26, 30, 50]:
        df[f'SMA_{w}'] = group['Close'].transform(lambda x: x.rolling(w, min_periods=1).mean())
    for w in [5, 12, 26, 50]:
        df[f'EMA_{w}'] = group['Close'].transform(lambda x: x.ewm(span=w, adjust=False).mean())

    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    def get_rsi(s, n=14):
        delta = s.diff()
        gain = delta.where(delta > 0, 0).rolling(n).mean()
        loss = -delta.where(delta < 0, 0).rolling(n).mean()
        return 100 - (100 / (1 + (gain/loss)))
    df['RSI'] = group['Close'].transform(get_rsi)

    df['Bollinger_Band_Middle'] = df['SMA_20']
    df['SD'] = group['Close'].transform(lambda x: x.rolling(20, min_periods=1).std())
    df['Bollinger_Band_Upper'] = df['Bollinger_Band_Middle'] + (2 * df['SD'])
    df['Bollinger_Band_Lower'] = df['Bollinger_Band_Middle'] - (2 * df['SD'])
    
    # OBV
    df['OBV'] = group.apply(lambda x: (np.sign(x['Close'].diff()).fillna(0) * x['Volume']).cumsum(), include_groups=False).reset_index(level=0, drop=True)

    # Weekly/Monthly Stats
    df['weekly_averages'] = group['Close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Weekly_SD'] = group['Close'].transform(lambda x: x.rolling(5, min_periods=1).std())
    df['Two_week_averages'] = group['Close'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df['Two_Week_SD'] = group['Close'].transform(lambda x: x.rolling(14, min_periods=1).std())
    df['Monthly_Average'] = group['Close'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df['Monthly_SD'] = group['Close'].transform(lambda x: x.rolling(30, min_periods=1).std())

    # 5. VIX
    print("Adding VIX...")
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)['Close'].reset_index()
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
        vix.columns = ['Date', 'VIX_Close']
        vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
        df = pd.merge(df, vix, on='Date', how='left')
    except:
        df['VIX_Close'] = np.nan

    # 6. TARGETS
    print("Generating targets...")
    df['next_day_pct_change'] = (group['Close'].shift(-1) - df['Close']) / df['Close']
    df['Movement'] = (df['next_day_pct_change'] > 0).astype(int)
    df['next_5_day_pct_change'] = (group['Close'].shift(-5) - df['Close']) / df['Close']
    df['Movement_5_day'] = (df['next_5_day_pct_change'] > 0).astype(int)
    df['next_30_day_pct_change'] = (group['Close'].shift(-30) - df['Close']) / df['Close']
    df['Movement_30_day'] = (df['next_30_day_pct_change'] > 0).astype(int)

    for col in ['earnings_bool', 'Split_Indicator']:
        df[col] = 0
    
    df.reset_index(drop=True).to_feather(final_feather_file)
    print(f"\n✅ SUCCESS: Saved to {final_feather_file}")

if __name__ == "__main__":
    run_sp500_pipeline("2024-01-01", "2024-12-01")