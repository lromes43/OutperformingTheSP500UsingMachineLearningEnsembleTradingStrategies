import pandas as pd
import yfinance as yf
import os
import time

url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
sp500 = pd.read_csv(url)
tickers = sp500['Symbol'].tolist()

os.makedirs("sp500_yearly_data", exist_ok=True)

for i, ticker in enumerate(tickers):
    file_path = f"sp500_yearly_data/{ticker}.csv"
    
    # Skip if already downloaded
    if os.path.exists(file_path):
        print(f"⏩ Skipping {ticker} ({i+1}/{len(tickers)}) — already exists")
        continue
    
    try:
        print(f"⬇️ Downloading {ticker} ({i+1}/{len(tickers)})...")
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty:
            print(f"⚠️ No data for {ticker}")
            continue
        
        data.to_csv(file_path)
        print(f"Saved {ticker} ({len(data)} rows)\n")
        
        time.sleep(0.5)  # slow down requests to avoid blocking
        
    except Exception as e:
        print(f" Error with {ticker}: {e}")
        time.sleep(5)  # wait longer before next one


import os
import pandas as pd

# Path to the parent directory containing all the folders
parent_dir = "p/Users/lukeromes/Desktop/Notre Dame/Mod2/Machine Learning/sp500_yearly_data"

# List to hold all dataframes
all_data = []

# Loop over each folder in the parent directory
for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)
    
    # Make sure it's a directory
    if os.path.isdir(folder_path):
        # Loop over each file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                
                # Read the CSV
                df = pd.read_csv(file_path)
                
                # Add filename as a new column (without extension if you want)
                df['source_file'] = os.path.splitext(file_name)[0]
                
                # Append to list
                all_data.append(df)

# Concatenate all dataframes into one
final_df = pd.concat(all_data, ignore_index=True)

# Optional: save to a new CSV
final_df.to_csv("combined_data.csv", index=False)

print("All CSVs have been combined successfully!")





import os
import pandas as pd

# Path to the folder containing all ticker CSV files
parent_dir = r "/Users/lukeromes/Desktop/Notre Dame/Mod2/Machine Learning/sp500_yearly_data"

all_data = []

for file_name in os.listdir(parent_dir):
    if file_name.endswith(".csv"):
        file_path = os.path.join(parent_dir, file_name)

        df = pd.read_csv(file_path)

        # Add ticker symbol
        ticker = os.path.splitext(file_name)[0]
        df['Ticker'] = ticker

        all_data.append(df)

# Combine all CSVs
final_df = pd.concat(all_data, ignore_index=True)

# Save combined file
final_df.to_csv("combined_data.csv", index=False)

print("Combined successfully!")
