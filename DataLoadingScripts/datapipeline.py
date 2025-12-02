

import pandas as pd
import yfinance as yf
import os
import time
from IPython.display import display, HTML


download_dir = "sp500_yearly_data"
url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
start_date = "2025-11-11"
end_date = "2025-12-01"
output_file = "combined_sp500_data.csv"


os.makedirs(download_dir, exist_ok=True)

try:
   
    sp500 = pd.read_csv(url)
    tickers = sp500['Symbol'].tolist()
    
    print(f"Starting download for {len(tickers)} S&P 500 companies...")
    print(f"Data period: Start Date = {start_date}, End Date = {end_date} (Daily Interval)\n")

except Exception as e:
    print(f"Error loading S&P 500 list: {e}")
    tickers = []



if tickers:
    for i, ticker in enumerate(tickers):
        file_path = os.path.join(download_dir, f"{ticker}.csv")
        if os.path.exists(file_path):
        try:
            print(f"⬇️ Downloading {ticker} ({i+1}/{len(tickers)})...")

            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                interval="1d", 
                progress=False
            )
            
            if data.empty:
                print(f" No data for {ticker}")
                continue

            data.to_csv(file_path)
         
            time.sleep(0.5)  
        except Exception as e:
            print(f" Error with {ticker}: {e}")
            time.sleep(5)  

    print("\n--- Download Phase Complete ---\n")



all_data = []
file_count = 0


for file_name in os.listdir(download_dir):
    if file_name.endswith(".csv"):
        file_path = os.path.join(download_dir, file_name)
        
        try:
            df = pd.read_csv(file_path)
            
       
            ticker_symbol = os.path.splitext(file_name)[0]
            df['Ticker'] = ticker_symbol
            
            
            df = df.rename(columns={'Date': 'Date'})
            
           
            desired_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
            df = df[desired_columns]
            
            all_data.append(df)
            file_count += 1
            
        except Exception as e:
            print(f"Could not read {file_name}: {e}")

if all_data:
    
    final_df = pd.concat(all_data, ignore_index=True)
 
    final_df.to_csv(output_file, index=False)

    print(f"\n--- Combination Phase Complete ---")
    print(f"Successfully combined data from {file_count} files.")
    print(f"Final columns: {list(final_df.columns)}")
    print(f"File saved to: **{output_file}** (Total Rows: {len(final_df)})")

    display(HTML("<hr><h4>Preview of Combined Data:</h4>"))
    display(final_df.head())
    
else:
    print("\n No data was found or read to combine.")




