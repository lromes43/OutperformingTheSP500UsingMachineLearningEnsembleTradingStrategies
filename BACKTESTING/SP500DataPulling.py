#Sp500 Data Pulling

import pandas as pd 
import yfinance as yf

trade_log_data = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/trade_log.csv")

dates = trade_log_data['Date'].unique()



tickers_list = ["^GSPC"]
start_date = dates[0]
end_date =  dates[-1]


sp500 = yf.download(tickers_list, start=start_date, end=end_date)

sp500.columns = sp500.columns.droplevel(1)

# 3. Reset the index to turn 'Date' into a column
sp500_data = sp500.reset_index()

sp500_data = sp500_data.to_csv("SP500testdata.csv", index=True)