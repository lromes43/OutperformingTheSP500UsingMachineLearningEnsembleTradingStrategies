
def initial_data_loading(length, frequency):
    import pandas as pd
    import yfinance as yf
    import os
    import time
    import sys



    url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
    DOWNLOAD_DIR = "sp500_yearly_data"
    COMBINED_FILE = "combined_sp500_yearly_data.csv"


    start_time = time.time() 


    try:
        print("Fetching S&P 500 ticker list...")
        sp500 = pd.read_csv(url)
        tickers = sp500['Symbol'].tolist()
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        print(f"Starting download for {len(tickers)} S&P 500 companies.\n")
    except Exception as e:
        print(f"Error loading S&P 500 list: {e}")
        sys.exit(1)


    print("--- Download Phase ---")
    for i, ticker in enumerate(tickers):
        file_path = os.path.join(DOWNLOAD_DIR, f"{ticker}.csv")
        
    
        if os.path.exists(file_path):
            print(f"⏩ Skipping {ticker} ({i+1}/{len(tickers)}) — already exists")
            continue
        
        try:
            print(f"⬇️ Downloading {ticker} ({i+1}/{len(tickers)})...", end='\r')
            
        
            data = yf.download(ticker, period="length", interval="frequency", progress=False)
            
            if data.empty:
                print(f"⚠️ No data for {ticker} (Possible delisting/missing data).")
                continue
            
            data.to_csv(file_path)
            print(f"Saved {ticker} ({len(data)} rows)                       \n")
            
            time.sleep(0.5)  
            
        except Exception as e:
            print(f" Error with {ticker}: {e}")
            time.sleep(5)  

    print("--- Download Phase Complete ---\n")


    print(f"--- Merging Data from '{DOWNLOAD_DIR}' ---")
    all_data = []
    file_count = 0


    for file_name in os.listdir(DOWNLOAD_DIR):
        if file_name.endswith(".csv"):
            file_path = os.path.join(DOWNLOAD_DIR, file_name)
            
            try:
            
                df = pd.read_csv(file_path)
                
            
                ticker = os.path.splitext(file_name)[0]
                df['Ticker'] = ticker
                
            
                all_data.append(df)
                file_count += 1
            except Exception as e:
                print(f"Error reading and processing {file_name}: {e}")


    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Save the combined file
        final_df.to_csv(COMBINED_FILE, index=False)
        
        print(f"✅ Successfully combined {file_count} CSVs.")
        print(f"Data saved to: {COMBINED_FILE}")
        print(f"Total rows in combined file: {len(final_df)}")
    else:
        print(" No CSV files were found to combine.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


        #NEW STUFFF CHECK!!!

    import pandas as pd
    import re

    date_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/combined_sp500_data.csv", header=None)


    clean = df[df[0].astype(str).str.match(date_regex)]

    clean.to_csv("cleaned.csv", index=False, header=False)


    import pandas as pd
    import re

    date_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    df = pd.read_csv("cleaned.csv", header=None)


    df = df[df[0].astype(str).str.match(date_regex)]

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]

    df.to_feather('FinalTestData.feather')

    return(final_df)



length = '1y'
frequency = '1d'

initial_data_loading(length=length, frequency=frequency)
