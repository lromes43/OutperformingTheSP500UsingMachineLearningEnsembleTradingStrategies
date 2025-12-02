import pandas as pd
import yfinance as yf
import os
import time
import sys

# --- Configuration ---
url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
DOWNLOAD_DIR = "sp500_yearly_data"
COMBINED_FILE = "combined_sp500_yearly_data.csv"
# --- ---

start_time = time.time() 

# 1. Setup and Get Tickers
try:
    print("Fetching S&P 500 ticker list...")
    sp500 = pd.read_csv(url)
    tickers = sp500['Symbol'].tolist()
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Starting download for {len(tickers)} S&P 500 companies.\n")
except Exception as e:
    print(f"Error loading S&P 500 list: {e}")
    sys.exit(1)

# 2. Download Phase
print("--- Download Phase ---")
for i, ticker in enumerate(tickers):
    file_path = os.path.join(DOWNLOAD_DIR, f"{ticker}.csv")
    
    # Skip if already downloaded
    if os.path.exists(file_path):
        print(f"⏩ Skipping {ticker} ({i+1}/{len(tickers)}) — already exists")
        continue
    
    try:
        print(f"⬇️ Downloading {ticker} ({i+1}/{len(tickers)})...", end='\r')
        
        # Download data for the last 1 year (period="1y")
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        if data.empty:
            print(f"⚠️ No data for {ticker} (Possible delisting/missing data).")
            continue
        
        data.to_csv(file_path)
        print(f"Saved {ticker} ({len(data)} rows)                       \n")
        
        time.sleep(0.5)  # slow down requests
        
    except Exception as e:
        print(f" Error with {ticker}: {e}")
        time.sleep(5)  # wait longer

print("--- Download Phase Complete ---\n")

# 3. Merging Phase (The Fix!)
print(f"--- Merging Data from '{DOWNLOAD_DIR}' ---")
all_data = []
file_count = 0

# The data is in the DOWNLOAD_DIR, so we iterate over files in that single folder.
for file_name in os.listdir(DOWNLOAD_DIR):
    if file_name.endswith(".csv"):
        file_path = os.path.join(DOWNLOAD_DIR, file_name)
        
        try:
            # Read the CSV
            df = pd.read_csv(file_path)
            
            # The Ticker is the filename without the .csv extension
            ticker = os.path.splitext(file_name)[0]
            df['Ticker'] = ticker
            
            # Append to list
            all_data.append(df)
            file_count += 1
        except Exception as e:
            print(f"Error reading and processing {file_name}: {e}")

# Concatenate all dataframes into one
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save the combined file
    final_df.to_csv(COMBINED_FILE, index=False)
    
    print(f"✅ Successfully combined {file_count} CSVs.")
    print(f"Data saved to: {COMBINED_FILE}")
    print(f"Total rows in combined file: {len(final_df)}")
else:
    print("❌ No CSV files were found to combine.")

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")