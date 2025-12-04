def creating_derived_vars_function (data):
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from datetime import datetime
    import time
    daily_return = []
    for i in range(0, len(data)):
        previous_price = data['Close'].iloc[i-1]
        current_price = data['Close'].iloc[i]
        daily_return_rate = ((current_price - previous_price)/previous_price) * 100
        daily_return.append(daily_return_rate)
    data['Daily_Return'] =  daily_return

    price_range = []
    for i in range(0, len(data)):
        high = data['High'].iloc[i]
        low = data['Low'].iloc[i]
        p_range = high - low
        price_range.append(p_range)
    data['Day_Range'] =   price_range

    data['SMA_5'] = data.groupby('Ticker')['Close'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop = True)
    data['SMA_12'] = data.groupby('Ticker')['Close'].rolling(window = 12, min_periods=1).mean().reset_index(level=0, drop=True)
    data['SMA_15'] = data.groupby('Ticker')['Close'].rolling(window=15, min_periods=1).mean().reset_index(level=0, drop = True)
    data['SMA_20'] = data.groupby('Ticker')['Close'].rolling(window = 20, min_periods=1).mean().reset_index(level = 0, drop=True) #made so can calc bollinger bands
    data['SMA_26'] = data.groupby('Ticker')['Close'].rolling(window=26, min_periods=1).mean().reset_index(level=0, drop=True)
    data['SMA_30'] = data.groupby('Ticker')['Close'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop = True)
    data['SMA_50'] = data.groupby('Ticker')['Close'].rolling(window=50, min_periods=1).mean().reset_index(level=0, drop = True)




    def ema_formula(price_current, ema_previous, N):
        K = 2/(N+1)
        return price_current * K + ema_previous * (1 - K)

    EMA_5 = []

    for ticker in data["Ticker"].unique():
        subset = data[data['Ticker'] == ticker].copy()
        individual_ema = []
        N = 5
        print(ticker)
        for i in range(0, len(subset)):
            print(i)
            price_current = subset['Close'].iloc[i]
            print(price_current)
            if  i == 0:
                ema_previous = subset['SMA_5'].iloc[0]
            else:
                ema_previous = EMA
            EMA = ema_formula(price_current, ema_previous, N)
            print(EMA)
            individual_ema.append(EMA)
        subset['EMA_5'] = individual_ema
        EMA_5.append(subset)

    data = pd.concat(EMA_5, ignore_index=True)


    EMA_12 = []

    for ticker in data["Ticker"].unique():
        subset = data[data['Ticker'] == ticker].copy()
        individual_ema = []
        N = 12
        print(ticker)
        for i in range(0, len(subset)):
            print(i)
            price_current = subset['Close'].iloc[i]
            print(price_current)
            if  i == 0:
                ema_previous = subset['SMA_12'].iloc[0]
            else:
                ema_previous = EMA
            EMA = ema_formula(price_current, ema_previous, N)
            print(EMA)
            individual_ema.append(EMA)
        subset['EMA_12'] = individual_ema
        EMA_12.append(subset)

    data= pd.concat(EMA_12, ignore_index=True)
    
    EMA_26 = []

    for ticker in data["Ticker"].unique():
        subset = data[data['Ticker'] == ticker].copy()
        individual_ema = []
        N = 26
        print(ticker)
        for i in range(0,len(subset)):
            print(i)
            price_current = subset['Close'].iloc[i]
            print(price_current)
            if i == 0:
                ema_previous = subset['SMA_26'].iloc[0]
            else:
                ema_previous = EMA
            EMA = ema_formula(price_current, ema_previous, N)
            print(EMA)
            individual_ema.append(EMA)
        subset['EMA_26'] = individual_ema
        EMA_26.append(subset)

    data = pd.concat(EMA_26, ignore_index=True)


    EMA_50 = []

    for ticker in data["Ticker"].unique():
        subset = data[data['Ticker'] == ticker].copy()
        individual_ema = []
        N = 50
        print(ticker)
        for i in range(0,len(subset)): 
            print(i)
            price_current = subset['Close'].iloc[i]
            if i == 0:
                ema_previous = subset['SMA_50'].iloc[0]
            else:
                ema_previous = EMA
            EMA = ema_formula(price_current, ema_previous, N)
            individual_ema.append(EMA)
        subset['EMA_50'] = individual_ema
        EMA_50.append(subset)

    data = pd.concat(EMA_50, ignore_index=True)



    MACD = []

    def macd_formula(EMA_12, EMA_26):
        return EMA_12 - EMA_26


    for ticker in data['Ticker'].unique():
        subset = data[data['Ticker'] == ticker].copy()
        individual_macd = []
        print(ticker)
        for i in range(0,len(subset)):
            print(i)
            EMA_12 = subset['EMA_12'].iloc[i]
            EMA_26 = subset['EMA_26'].iloc[i]
            MACD_Vals = macd_formula(EMA_12, EMA_26)
            individual_macd.append(MACD_Vals)
        subset['MACD'] = individual_macd
        MACD.append(subset)

    data = pd.concat(MACD, ignore_index=True)


    RSI = []
    N = 14

    for ticker in data['Ticker'].unique():
        U = []
        D = []
        individual_rsi = []
        subset = data[data['Ticker'] == ticker].copy()
        for i in range(0, len(subset)):
            change = subset['Close'].iloc[i] - subset['Close'].iloc[i -1 ]
            if change > 0: 
                U.append(change)
                D.append(0)
            else:
                U.append(0)
                D.append(abs(change))
            AU = np.sum(U) / N
            AD = np.sum(D) / N
            if AD == 0:
                RS = 100
            else: 
                RS = AU/AD
            RSI_Vals = 100 - (100 / (1 + RS))
            individual_rsi.append(RSI_Vals)
        subset['RSI'] = individual_rsi
        RSI.append(subset)

    data = pd.concat(RSI, ignore_index=True)

    #Bollinger Bamnds

    #Middle Band
    
    data['Bollinger_Band_Middle'] = data['SMA_20']

    #Calculates the SD for each
    std_vals = data.groupby('Ticker')['Bollinger_Band_Middle'].std()
    std_vals = std_vals.reset_index(name='SD')

    #Upper Band
    SD_merge = pd.merge(data, std_vals, how = 'left', on = 'Ticker')
    data = SD_merge
    data.columns
    data['Bollinger_Band_Upper'] = data['SMA_20'] +  2* data['SD']
    data['Bollinger_Band_Lower'] = data['SMA_20'] -  2* data['SD']


    #On Balance Volume


    OBV = []
    for ticker in data['Ticker'].unique():
        subset = data[data['Ticker'] == ticker].copy()
        OBV_individual = []
        for i in range(len(subset)):
            if i == 0: 
                OBV_individual.append(0)
                continue

            close_change = subset['Close'].iloc[i] - subset['Close'].iloc[i-1]
            if close_change > 0: 
                addend = subset['Volume'].iloc[i]
            elif close_change < 0:
                addend = -(subset['Volume'].iloc[i])
            else: 
                addend = 0
        
            OBV_prev = OBV_individual[-1]
            OBV_val = OBV_prev + addend
            OBV_individual.append(OBV_val)
        subset['OBV'] = OBV_individual
        OBV.append(subset)
    data = pd.concat(OBV, ignore_index=True)

    #VIX

    ticker_symbol = "^VIX"
    start_date = "2025-11-10"
    end_date = "2025-12-01"

    try:
        vix_data = yf.download(
            tickers=ticker_symbol,
            start=start_date,
            end=end_date,
            interval="1d" 
        )

        vix_close_prices = vix_data['Close']

        print("VIX Closing Prices from {} to {}:".format(start_date, end_date))
        print(vix_close_prices)

    except Exception as e:
        print(f"An error occurred: {e}")

    vix = vix_close_prices.reset_index()
    vix.columns = ["Date", "VIX_Close"]
    vix["Date"] = pd.to_datetime(vix["Date"])
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.merge(vix, on="Date", how="left")


    #earnings data


    url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
    sp500 = pd.read_csv(url)
    tickers = sp500["Symbol"].tolist()

    cols = ['Ticker','date','EPS Estimate','Reported EPS','Surprise(%)']
    all_earnings = pd.DataFrame(columns=cols)

    start_date = datetime.strptime("2025-11-10", "%Y-%m-%d").date()
    end_date   = datetime.strptime("2025-12-01", "%Y-%m-%d").date()


    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Getting {t}...")

        try:
            df = yf.Ticker(t).get_earnings_dates(limit=50)
        except Exception as e:
            print(f"    Error for {t}: {e}")
            continue
        if df is None or len(df) == 0:
            continue

        df.index = pd.to_datetime(df.index).date
        df = df.reset_index().rename(columns={"index": "date"})

        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            continue

        df["Ticker"] = t
        all_earnings = pd.concat([all_earnings, df], ignore_index=True)

        time.sleep(0.25)  


    all_earnings.columns = ['Ticker', 'Date', 'EPS Estimate', 'Reported EPS', 'Surprise(%)']
    all_earnings


    original_data = data
    earnings_data = all_earnings

    original_data['Earnings_Date'] = [None] * len(original_data)
    og = original_data.sort_values(by = ['Ticker','Date'])
    og = og.drop('Earnings_Date', axis = 1)
    e = earnings_data.sort_values(by = ['Ticker', 'Date'])

    og['Date'] = pd.to_datetime(og['Date'])
    e['Date'] = pd.to_datetime(e['Date'])

    data['Date'] = pd.to_datetime(data['Date'])

    merged_df = pd.merge(left = og, 
                     right = e,
                     on = ['Date', 'Ticker'], 
                     how = 'left')
    merged_df['earnings_bool'] = np.where(merged_df['EPS Estimate'].notna(), 1, 0)

    data = merged_df

    data = data.copy()
    
    #weekly averages
  
    data['weekly_averages'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    data['Weekly_SD'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(5, min_periods=1).std())

    #two week averages

    data['Two_week_averages'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    data['Two_Week_SD'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(14, min_periods=1).std())

    #month average

    data['Monthly_Average'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    data['Monthly_SD'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(30, min_periods=1).std())

    #stock splits

    tickers_list = data['Ticker'].unique().tolist() 
    tickers = tickers_list 

    start = "2025-11-10"
    end = "2025-12-01"

    all_splits = []

    for t in tickers:
        ticker_splits = yf.Ticker(t).splits 
        
        if ticker_splits is not None and len(ticker_splits) > 0:
            temp = ticker_splits[(ticker_splits.index >= start) & (ticker_splits.index <= end)]
            for date, ratio in temp.items():
                all_splits.append({"Ticker": t, "Date": date, "Split Ratio": ratio})
                print(t, date, ratio)

    df = pd.DataFrame(all_splits)
  
    split = df
  
    split['Date'] = pd.to_datetime(split['Date'], utc=True)
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    merged_df = pd.merge(data, split, how = 'left', left_on=['Date', 'Ticker'], right_on=['Date', 'Ticker'])


    Split = []
    for i in range(0, len(merged_df)):
        current_value = merged_df['Split Ratio'].iloc[i]
        if not pd.isna(current_value): 
            Split.append(1)
        else:
            Split.append(0)

    merged_df['Split_Indicator'] = Split

    data = merged_df

    #Reordering


    cols_reordered = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Day_Range', 
        'SMA_5', 'SMA_12', 'SMA_15', 'SMA_20', 'SMA_26', 'SMA_30', 'SMA_50', 
        'EMA_5', 'EMA_12', 'EMA_26', 'EMA_50', 'MACD', 'RSI', 'Bollinger_Band_Middle', 
        'SD', 'Bollinger_Band_Upper', 'Bollinger_Band_Lower', 'OBV',
        'EPS Estimate', 'Reported EPS', 'Surprise(%)', 'earnings_bool', 
        'Split Ratio', 'Split_Indicator', 'Daily_Return', 
        'weekly_averages', 'Two_week_averages', 'Monthly_Average', 
        'Weekly_SD', 'Two_Week_SD', 'Monthly_SD'
        ]
    data = data.loc[:, cols_reordered]


    Day_to_Day_Diff = []

    for ticker in data['Ticker'].unique():
        subset = data[data['Ticker'] == ticker].copy()
        subset = subset.sort_values('Date')

        individual_d2d_diff = []

        for i in range(0, len(subset)):
            if i == len(subset)-1:
                diff_val = None     # last day has no next-day label
            else:
                price_today = subset['Close'].iloc[i]
                price_tom = subset['Close'].iloc[i+1]
                diff_val = (price_tom - price_today) / price_today  # percent change
            individual_d2d_diff.append(diff_val)

        subset['next_day_pct_change'] = individual_d2d_diff
        Day_to_Day_Diff.append(subset)

    data = pd.concat(Day_to_Day_Diff, ignore_index=True)

    data['Movement'] = (data['next_day_pct_change'] > 0).astype(int)

    data = data.sort_values(['Ticker', 'Date'])

    data['next_5_day_pct_change'] = (
        data.groupby('Ticker')['Close'].shift(-5) - data['Close']
    ) / data['Close']

    data['Movement_5_day'] = (data['next_5_day_pct_change'] > 0).astype(int)
    data = data.sort_values(['Ticker', 'Date'])

    data['next_30_day_pct_change'] = (
        data.groupby('Ticker')['Close'].shift(-30) - data['Close']
    ) / data['Close']

    data['Movement_30_day'] = (data['next_30_day_pct_change'] > 0).astype(int)
    final_data = data
    final_data.to_feather('CHECKFEATHER.feather')

    final_data 




#Calling Creating Vars Function to create derived vars


import pandas as pd
data = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/cleaned.csv")
test = creating_derived_vars_function(data)
