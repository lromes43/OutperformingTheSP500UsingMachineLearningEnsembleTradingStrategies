

def initial_data_loading_pipeline (start_date, end_date,columns, download_dir, final_output_file):

    import pandas as pd
    import yfinance as yf
    import os
    import time
    import sys
    import re

    download_dir = download_dir
    url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
    output_file = "test_sp500_data.csv" 
    final_output_file = final_output_file 
    start_date = start_date
    end_date = end_date
    desired_columns = columns

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
                print(f"Downloading {ticker} ({i+1}/{len(tickers)})...", end='\r')
                
            
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


    folder_path =  download_dir
    all_data = []

    

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            
            df = pd.read_csv(file_path)
            
            df['source_file'] = os.path.splitext(file_name)[0]
            
            all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    
    df = final_df
    print(df) #good up to here

    print("All CSVs have been combined successfully!")

    date_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")


    clean = df[df[0].astype(str).str.match(date_regex)]
    df = clean

    date_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    df = df[df[0].astype(str).str.match(date_regex)]

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]

    df.to_feather('FinalTestData.feather')

    return df




start_date = "2024-11-11"
end_date = "2025-11-10"
download_dir = "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/DataPipelineData"
final_output_file = "combined_sp500_data.csv" 
columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']



initial_data_loading_pipeline(start_date=start_date, end_date=end_date,columns=columns, download_dir=download_dir, final_output_file=final_output_file)


