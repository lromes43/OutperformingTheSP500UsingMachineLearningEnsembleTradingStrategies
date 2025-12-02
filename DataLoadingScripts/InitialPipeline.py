import os
import time
import re
import sys
import pandas as pd
import yfinance as yf

def initial_data_loading_pipeline(start_date, end_date, columns, download_dir, final_output_file):
    os.makedirs(download_dir, exist_ok=True)

    # ---- Load S&P 500 list ----
    print("Fetching S&P 500 ticker list...")
    url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
    try:
        sp500 = pd.read_csv(url)
        tickers = sp500["Symbol"].tolist()
    except Exception as e:
        print(f"Error loading S&P 500 list: {e}")
        sys.exit(1)

    print(f"Starting download for {len(tickers)} companies...\n")

    # ---- Helper to fix broken CSVs ----
    def fix_csv(path):
        with open(path, "r") as f:
            lines = f.read().splitlines()

        if len(lines) > 2 and lines[1].startswith("Ticker,") and lines[2].startswith("Date"):
            print(f"Fixing: {os.path.basename(path)}")
            header = lines[0].split(",")
            data_lines = lines[3:]
            clean_csv = "Date," + ",".join(header[1:]) + "\n" + "\n".join(data_lines)
            with open(path, "w") as f:
                f.write(clean_csv)
            return True
        return False

    # ---- Download loop ----
    failed_tickers = []
    for ticker in tickers:
        file_path = os.path.join(download_dir, f"{ticker}.csv")
        if os.path.exists(file_path):
            continue

        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False
            )

            # Only save if data exists and covers the start_date
            if not data.empty and data.index.min() <= pd.to_datetime(start_date):
                data.to_csv(file_path)
            else:
                failed_tickers.append(ticker)

            time.sleep(0.5)

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            failed_tickers.append(ticker)
            time.sleep(5)

    print(f"\nDownload complete. Failed tickers: {failed_tickers}\n")

    # ---- Fix CSVs ----
    fixed = 0
    for file in os.listdir(download_dir):
        if file.endswith(".csv"):
            full_path = os.path.join(download_dir, file)
            if fix_csv(full_path):
                fixed += 1
    print(f"Cleaned {fixed} CSV files.\n")

    # ---- Combine CSVs ----
    all_dfs = []
    for file in os.listdir(download_dir):
        if file.endswith(".csv"):
            path = os.path.join(download_dir, file)
            df = pd.read_csv(path)

            if df.empty or "Date" not in df.columns:
                continue

            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            df["Ticker"] = os.path.splitext(file)[0]
            all_dfs.append(df)

    if len(all_dfs) == 0:
        print("No valid CSVs found. Exiting.")
        sys.exit(1)

    final_df = pd.concat(all_dfs, ignore_index=True)

    # ---- Clean Dates ----
    date_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    final_df = final_df[final_df["Date"].astype(str).str.match(date_regex)]

    # ---- Select and reorder columns ----
    final_df = final_df[columns]

    # ---- Save to feather or CSV ----
    final_df.to_feather("FinalTestData.feather")
    final_df.to_csv(final_output_file, index=False)
    print(f"Saved combined dataset to {final_output_file}")

    return final_df


# ---- Run the pipeline ----
start_date = "2024-11-11"
end_date = "2025-11-10"
download_dir = "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/DataPipelineData"
final_output_file = "combined_sp500_data.csv"
columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']

df = initial_data_loading_pipeline(
    start_date=start_date,
    end_date=end_date,
    columns=columns,
    download_dir=download_dir,
    final_output_file=final_output_file
)
