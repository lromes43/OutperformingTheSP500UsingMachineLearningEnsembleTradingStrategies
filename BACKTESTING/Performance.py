import pandas as pd
from plotnine import *
import yfinance as yf
import statsmodels.formula.api as smf

data = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/trade_log.csv")

buy = data[data['Action'] == "BUY"]
sell = data[data['Action'] == "SELL"]

buy = buy.rename(columns={'Date': 'Buy_Date', 'Price': 'Buy_Price'}).drop('Action', axis = 1)
sell = sell.rename(columns={'Date': 'Sell_Date', 'Price': 'Sell_Price'}).drop('Action', axis = 1)

merged = pd.merge(buy, sell, how = 'inner', on = 'Ticker')

new_order= ['Ticker', 'Buy_Date', 'Sell_Date', 'Buy_Price', 'Sell_Price']
merged = merged[new_order]

merged['P/L per share'] = merged['Sell_Price'] - merged['Buy_Price']
merged['Trade Return Pct'] = (merged['Sell_Price'] - merged['Buy_Price']) / merged['Buy_Price'] * 100


total_p = float(merged.loc[merged['P/L per share'] >=0, 'P/L per share'].sum())

total_l = float(abs((merged.loc[merged['P/L per share'] < 0, 'P/L per share'].sum())))

profit_factor = total_p / total_l

#1.075505404977202 profit factor

#Expectancy

#% of winners

winner_pct = float(merged.loc[merged['P/L per share'] > 0, 'P/L per share'].count() / len(merged))
#56.324

loser_pct = float(merged.loc[merged['P/L per share'] < 0, 'P/L per share'].count() / len(merged))
43.676

winner_avg_val = float(merged.loc[merged['P/L per share'] > 0, 'P/L per share'].sum() / merged.loc[merged['P/L per share'] > 0, 'P/L per share'].count() )
4.896
loser_avg_val = float(merged.loc[merged['P/L per share'] < 0, 'P/L per share'].sum() / merged.loc[merged['P/L per share'] < 0, 'P/L per share'].count() )
-5.87


Expectancy = (winner_pct * winner_avg_val) + (loser_pct *loser_avg_val)


#Trade returns histogram 


plot = ggplot(merged, aes(x = 'Trade Return Pct')) + geom_histogram(binwidth = 1, 
                                                                    fill = 'dodgerblue',
                                                                    color = 'black', alpha = .3
                                                                    )

plot.show()

#many trades near 0 
#winners have more room to run and losers have more room to fall

#median of returns 

median = float(merged['Trade Return Pct'].median())
# median is .466
mean = float(merged['Trade Return Pct'].mean())
#mean is .560

#mean is greater than median so following trend

kurtosis = merged['Trade Return Pct'].kurtosis()
kurtosis

skew = merged['Trade Return Pct'].skew()
print(skew)


#Regressing against market factors 
#VIX

trade_log_data = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/trade_log.csv")
dates =data['Date'].unique()
tickers_list = ["^VIX"]
start_date = dates[0]
end_date =  dates[-1]
VIX = yf.download(tickers_list, start=start_date, end=end_date)

VIX.columns = VIX.columns.droplevel(1)

VIX_data = VIX.reset_index()

VIX_data['Date'] = pd.to_datetime(VIX_data['Date'])
VIX_data = VIX_data.drop(['High', 'Low', 'Open', 'Volume'], axis=1)
VIX_data = VIX_data.rename(columns={'Close': 'VIX_Close'})
merged['Sell_Date'] = pd.to_datetime(merged['Sell_Date'])

merged_final = pd.merge(merged, VIX_data, how = 'inner', left_on='Sell_Date', right_on='Date')


tickers_list = ["MTUM"]
start_date = dates[0]
end_date =  dates[-1]


MTUM = yf.download(tickers_list, start=start_date, end=end_date)

MTUM.columns = MTUM.columns.droplevel(1)

MTUM_data = MTUM.reset_index()


MTUM_data ['Date'] = pd.to_datetime(MTUM_data ['Date'])
MTUM_data =MTUM_data.drop(['High', 'Low', 'Open', 'Volume'], axis=1)
MTUM_data  = MTUM_data .rename(columns={'Close': 'MTUM_Close'})


merged_final = pd.merge(merged_final, MTUM_data , how = 'inner', left_on='Sell_Date', right_on='Date')

merged_final = merged_final.rename(columns={'Trade Return Pct': 'trade_return_pct','VIX_Close': 'vix_close'})

VIXmodel= smf.ols(formula='trade_return_pct ~ vix_close',data=merged_final).fit()
VIXmodel.summary()

#R-squared is .036 meaning strategy is uncorrelated with market fear. 
#only 3.6% trades motivated by market fear
# is diversifier that can be successful even if market is volatile

MTUMmodel = smf.ols(formula='trade_return_pct ~  MTUM_Close',data=merged_final).fit()
MTUMmodel.summary()

#R-squared is .024 meaning strategy is uncorrelated with market fear. 
#only 2.4% trades motivated by market fear

combinedmodel = smf.ols(formula='trade_return_pct ~ vix_close + MTUM_Close', data=merged_final).fit()
combinedmodel.summary()



def get_streaks(returns):
    is_win = returns > 0
    streaks = is_win.ne(is_win.shift()).cumsum()
    return is_win.groupby(streaks).sum()

win_streaks = get_streaks(merged['Trade Return Pct'])
print(f"Max Win Streak: {win_streaks.max()}")
print(f"Average Win Streak: {win_streaks.mean():.2f}")



cum_ret = (1 + merged['Trade Return Pct']/100).cumprod()
peak = cum_ret.cummax()
drawdown = (cum_ret - peak) / peak


is_underwater = drawdown < 0
runs = is_underwater.ne(is_underwater.shift()).cumsum()
max_drawdown_duration = is_underwater.groupby(runs).sum().max()

print(f"Longest Flat Period: {max_drawdown_duration} trades")

