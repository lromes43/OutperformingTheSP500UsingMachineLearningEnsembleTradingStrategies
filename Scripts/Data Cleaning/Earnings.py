import pandas as pd
import yfinance as yf
from datetime import datetime
import time

url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
sp500 = pd.read_csv(url)
tickers = sp500["Symbol"].tolist()

cols = ['Ticker','date','EPS Estimate','Reported EPS','Surprise(%)']
all_earnings = pd.DataFrame(columns=cols)

start_date = datetime.strptime("2024-11-11", "%Y-%m-%d").date()
end_date   = datetime.strptime("2025-11-10", "%Y-%m-%d").date()


for i, t in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] Getting {t}...")

    try:
        df = yf.Ticker(t).get_earnings_dates(limit=50)
    except Exception as e:
        print(f"    Error for {t}: {e}")
        continue

    # If ticker has no earnings data
    if df is None or len(df) == 0:
        continue

    # Convert to date column
    df.index = pd.to_datetime(df.index).date
    df = df.reset_index().rename(columns={"index": "date"})

    # Filter by date range
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    if df.empty:
        continue

    df["Ticker"] = t
    all_earnings = pd.concat([all_earnings, df], ignore_index=True)

    time.sleep(0.25)  


all_earnings.columns = ['Ticker', 'Date', 'EPS Estimate', 'Reported EPS', 'Surprise(%)']
all_earnings.to_csv('earnings.csv', index=False)



data = pd.read_csv("Data/FINALDATA.csv")
data










