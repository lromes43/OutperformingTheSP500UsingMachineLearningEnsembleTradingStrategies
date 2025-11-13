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


#Making SMA for SMA analysis and for initial EMA

data['SMA_5'] = data.groupby('Ticker')['Close'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop = True)
data['SMA_12'] = data.groupby('Ticker')['Close'].rolling(window = 12, min_periods=1).mean().reset_index(level=0, drop=True)
data['SMA_15'] = data.groupby('Ticker')['Close'].rolling(window=15, min_periods=1).mean().reset_index(level=0, drop = True)
data['SMA_26'] = data.groupby('Ticker')['Close'].rolling(window=26, min_periods=1).mean().reset_index(level=0, drop=True)
data['SMA_30'] = data.groupby('Ticker')['Close'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop = True)
data['SMA_50'] = data.groupby('Ticker')['Close'].rolling(window=50, min_periods=1).mean().reset_index(level=0, drop = True)


data

##Manual WAY!

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



#EMA
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
data

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
data

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
data

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
data



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
data.to_csv("DataWithSomeDerivedVars", index= False)


##RSI

#14 Day RSI

#Formula: 100 - (100 / (1 + AVG UP / AVG Down)

RSI = []
rsi_period = 14
period = int(rsi_period - 1)
subset = data[data['Ticker'] == 'CSCO']
subset2 = subset.iloc[0:rsi_period]
subset2

for i in range(0, len(subset2)):
    U = []
    D = []
    if subset2['Close'].iloc[i] - subset2['Close'].iloc[i-1] > 0:
        up = subset2['Close'].iloc[i] - subset2['Close'].iloc[i-1]
        U.append(up)
    else:
        up = 0
        U.append(up)
    if subset2['Close'].iloc[i] - subset2['Close'].iloc[i-1] < 0:
        down = abs(subset2['Close'].iloc[i] - subset2['Close'].iloc[i-1])
        D.append(down)
    else:
        down =0
        D.append(down)
    average_up = np.sum(U) / rsi_period
    average_down = np.sum(D) / rsi_period
    Relative_Strength = average_up / average_down
    RSI = 100 - (100 / (1 + Relative_Strength))
    






