import pandas as pd
import yfinance as yf
import numpy as np
data = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/finalsp500.csv")
data.head()

#Making a variable for current day price - previous price
daily_return = []
for i in range(0, len(data)):
    previous_price = data['Close'].iloc[i-1]
    current_price = data['Close'].iloc[i]
    daily_return_rate = ((current_price - previous_price)/previous_price) * 100
    daily_return.append(daily_return_rate)

data['Daily_Return'] =  [None] + daily_return
data

#Making variable to calculate intraday range
price_range = []
for i in range(0, len(data)):
    high = data['High'].iloc[i]
    low = data['Low'].iloc[i]
    p_range = high - low
    price_range.append(p_range)

price_range
data['Day_Range'] =  [None] + price_range
data


#Making variable to calculate 5 DAY SMA for each ticker

data['SMA_5'] = data.groupby('Ticker')['Close'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop = True)
data

#Making variable to calculate 15 DAY SMA for each ticker

data['SMA_15'] = data.groupby('Ticker')['Close'].rolling(window=15, min_periods=1).mean().reset_index(level=0, drop = True)
data
#Making variable to calculate 15 DAY SMA for each ticker

data['SMA_30'] = data.groupby('Ticker')['Close'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop = True)
data

data['SMA_50'] = data.groupby('Ticker')['Close'].rolling(window=50, min_periods=1).mean().reset_index(level=0, drop = True)
data

X = data.groupby('Ticker')['Date','Close']


#Exponential Moving Averages (EMAS)

data_encoded = pd.get_dummies(data, columns=['Ticker'])
data_encoded
data_encoded.columns.to_list()


#want to find EMA for N = 50,26,12, and 5
#EMA formula:  Price current  * (2/N+1) +EMA previous * ( 1- (2/N+1)) 

ticker_list = data['Ticker'].unique() 

subset = data[data['Ticker'] == "CSCO"].copy()

price_current = subset['Close'].iloc[0]

ema_previous = subset['SMA_50'].iloc[0]

N = 50
K = (2/N+1)

first_row_index = subset.index[0]
subset.loc[first_row_index, 'EMA_50'] = price_current * K + ema_previous * (1 - K)

subset


#gets all EMA 50 of CSCO ticket
EMA_50 =[]
target_tickers = ["CSCO", "PLL"]
subset = data[data['Ticker'].isin(target_tickers)].copy()

for i in range(1,len(subset)):
    price_current = subset['Close'].iloc[i]
    ema_previous = subset['SMA_50'].iloc[i]
    N = 50
    K = (2/N+1)
    EMA = price_current * K + ema_previous * (1 - K)
    EMA_50.append(EMA)


subset['EMA_50'] = [None] + EMA_50
subset['Ticker']





#to iterate through everything do we need to iterate subsetting?



for ticker in ticker_list:
    ticker_subset = data[data['Ticker'] == ticker].copy()
    EMA_50_Final =[]
    for i in range(0,len(ticker_subset)):
        price_current = ticker_subset['Close'].iloc[i]
        ema_previous = ticker_subset['SMA_50'].iloc[i]
        N = 50
        K = (2/N+1)
        EMA = price_current * K + ema_previous * (1 - K)
        EMA_50_Final.append(EMA) 


EMA_50_Final






            





        
    