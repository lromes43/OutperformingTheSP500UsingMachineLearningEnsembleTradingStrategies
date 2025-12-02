import pandas as pd
import yfinance as yf
import os
import time
import sys


download_dir = "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/DataPipelineData"
url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
output_file = "test_sp500_data.csv"


start_date = "2025-11-11"
end_date = "2025-12-01"


desired_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']

start_time = time.time() 


os.makedirs(download_dir, exist_ok=True)

try:
   
    print("Fetching S&P 500 ticker list...")
    sp500 = pd.read_csv(url)
    tickers = sp500['Symbol'].tolist()
    
    print(f"Starting download for {len(tickers)} S&P 500 companies.")
    print(f"Data period: {start_date} to {end_date} (Daily Interval)\n")

except Exception as e:
    print(f"Error loading S&P 500 list: {e}")
    sys.exit(1)



if tickers:
    for i, ticker in enumerate(tickers):
        file_path = os.path.join(download_dir, f"{ticker}.csv")
        

        if os.path.exists(file_path):
            continue
        
        try:
            print(f"⬇️ Downloading {ticker} ({i+1}/{len(tickers)})...", end='\r')
            
           
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                interval="1d", 
                progress=False
            )
            
            if data.empty:
                continue
            
         
            data.to_csv(file_path)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f" Error with {ticker}: {e}")
            time.sleep(5) 

    print("\n--- Download Phase Complete ---\n")



folder_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/DataPipelineData"



all_data = []


for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        
      
        df = pd.read_csv(file_path)
        
    
        df['source_file'] = os.path.splitext(file_name)[0]
        
   
        all_data.append(df)


final_df = pd.concat(all_data, ignore_index=True)


final_df.to_csv("combined_data.csv", index=False)

print("All CSVs have been combined successfully!")
