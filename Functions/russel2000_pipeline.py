import os
import pandas as pd
import yfinance as yf
import numpy as np
import time
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def micro_pipeline(start_date, end_date):
    BASE_DIR = "/Users/lukeromes/Desktop/Personal/Sp500Project"
    DATA_DIR = os.path.join(BASE_DIR, "Data")
    FAILED_FILE = os.path.join(DATA_DIR, "failed_tickers.csv")
    INPUT_FILE = "/Users/lukeromes/Desktop/Sp500Project/Data/Russell/russell_2000.csv"

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find input file at {INPUT_FILE}")
        return

    raw_data = pd.read_csv(INPUT_FILE)
    raw_tickers = raw_data['Ticker'].dropna().astype(str).unique()
    
    tickers = []
    metadata_keywords = ["BARCHART", "DOWNLOADED", "DATE", "TIME", "CST", "FROM"]
    
    for t in raw_tickers:
        t_clean = t.strip().upper()
        if any(key in t_clean for key in metadata_keywords):
            continue
        tickers.append(t_clean.replace('.', '-'))

    all_data = []
    failed = []

    print(f"Starting download for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers):
        try:
            df_temp = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if df_temp is None or df_temp.empty:
                failed.append((ticker, "Empty DataFrame"))
                continue

            if isinstance(df_temp.columns, pd.MultiIndex):
                df_temp.columns = df_temp.columns.get_level_values(0)

            df_temp = df_temp.reset_index()
            df_temp['Ticker'] = ticker
            all_data.append(df_temp)

        except Exception as e:
            failed.append((ticker, str(e)))

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(tickers)}")
        
        time.sleep(random.uniform(0.1, 0.3))

    if not all_data:
        print("No data was successfully downloaded.")
        return


    df = pd.concat(all_data, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values(['Ticker', 'Date']).drop_duplicates(subset=['Ticker', 'Date'])

    group = df.groupby('Ticker')

    df['Daily_Return'] = group['Close'].pct_change() * 100
    df['Day_Range'] = df['High'] - df['Low']

    for w in [5, 12, 15, 20, 26, 30, 50]:
        df[f'SMA_{w}'] = group['Close'].transform(lambda x: x.rolling(w, min_periods=1).mean())

    for w in [5, 12, 26, 50]:
        df[f'EMA_{w}'] = group['Close'].transform(lambda x: x.ewm(span=w, adjust=False).mean())

    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # RSI
    def get_rsi(s, n=14):
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df['RSI'] = group['Close'].transform(get_rsi)

    # Bollinger Bands
    df['SD'] = group['Close'].transform(lambda x: x.rolling(20, min_periods=1).std())
    df['Bollinger_Band_Middle'] = df['SMA_20']
    df['Bollinger_Band_Upper'] = df['Bollinger_Band_Middle'] + (2 * df['SD'])
    df['Bollinger_Band_Lower'] = df['Bollinger_Band_Middle'] - (2 * df['SD'])

    # OBV
    def calc_obv(x):
        return (np.sign(x['Close'].diff()).fillna(0) * x['Volume']).cumsum()
    
    df['OBV'] = group.apply(lambda x: calc_obv(x), include_groups=False).reset_index(level=0, drop=True)

    #VIX
    try:
        vix_raw = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = vix_raw.columns.get_level_values(0)
        vix = vix_raw['Close'].reset_index().rename(columns={'Close': 'VIX_Close'})
        vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
        df = df.merge(vix, on='Date', how='left')
    except:
        print("Warning: VIX data could not be merged.")


    df['next_day_close'] = group['Close'].shift(-1)
    df['next_5_day_close'] = group['Close'].shift(-5)
    df['next_30_day_close'] = group['Close'].shift(-30)

    df['next_day_pct_change'] = (df['next_day_close'] - df['Close']) / df['Close'] * 100
    df['next_5_day_pct_change'] = (df['next_5_day_close'] - df['Close']) / df['Close'] * 100
    df['next_30_day_pct_change'] = (df['next_30_day_close'] - df['Close']) / df['Close'] * 100

    df['Movement'] = (df['next_day_pct_change'] > 0).astype(int)
    df['Movement_5_day'] = (df['next_5_day_pct_change'] > 0).astype(int)
    df['Movement_30_day'] = (df['next_30_day_pct_change'] > 0).astype(int)


    df_1d = df.reset_index(drop=True)
    df_1d.to_feather(os.path.join(DATA_DIR, "FinalRUSSEL2000.feather"))

    pd.DataFrame(failed, columns=['Ticker', 'Error']).to_csv(FAILED_FILE, index=False)

    print(f"SUCCESS — Processed {len(df_1d):,} rows for 1D analysis (includes latest trading day).")
