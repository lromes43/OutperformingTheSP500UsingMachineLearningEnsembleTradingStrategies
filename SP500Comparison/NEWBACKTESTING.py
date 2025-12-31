import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from plotnine import *

DATA_PATH = "/Users/lukeromes/Desktop/Sp500Project/Data/FINALDATA.FEATHER"
TEST_DATA_PATH = "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/TestData.feather"
BINARY_MODEL_PATH = "/Users/lukeromes/Desktop/Sp500Project/Initial Models/FinalBoostedOneDayClassifier.joblib"
CONT_MODEL_PATH = "/Users/lukeromes/Desktop/Sp500Project/Initial Models/ContinuousOneDayFinal.job.lib"

print("Initializing Simulation...")
if not os.path.exists(BINARY_MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {BINARY_MODEL_PATH}")

binary_model = joblib.load(BINARY_MODEL_PATH)
cont_model = joblib.load(CONT_MODEL_PATH)
data = pd.read_feather(DATA_PATH)
test_data = pd.read_feather(TEST_DATA_PATH)

data['Date'] = pd.to_datetime(data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])


def get_predictions(date, model_bin, model_cont):
    """Generates a filtered list of stock picks for a given date."""
    subset = test_data[test_data['Date'] == date].copy()
    
    cols = ['ticker', 'prob', 'val']
    if subset.empty:
        return pd.DataFrame(columns=cols)

    drop_cols = ['Date', 'next_day_pct_change','Daily_Return','next_5_day_pct_change',
                 'Movement_5_day','next_30_day_pct_change','Movement_30_day',
                 'Movement', 'Ticker']
    
    X = subset.drop(columns=[c for c in drop_cols if c in subset.columns], errors='ignore')
    X_final = pd.get_dummies(X, drop_first=True)
    
    X_bin = X_final.reindex(columns=model_bin.feature_names, fill_value=0) if hasattr(model_bin, 'feature_names') else X_final
    X_cont = X_final.reindex(columns=model_cont.feature_names, fill_value=0)
    
    d_bin = xgb.DMatrix(X_bin)
    d_cont = xgb.DMatrix(X_cont)

    prob_up = model_bin.predict(d_bin)
    pred_val = model_cont.predict(d_cont)
    
    res = pd.DataFrame({
        'ticker': subset['Ticker'].values,
        'prob': prob_up,
        'val': pred_val * 100
    })
    
   
    filtered = res[(res['prob'] >= 0.5) & (res['val'] > 0)]
    return filtered if not filtered.empty else pd.DataFrame(columns=cols)

all_dates = sorted(data['Date'].unique())
dates = [d for d in all_dates if d > pd.Timestamp('2025-11-12')]

initial_capital = 100000.0
cash = initial_capital
per_position = 10000.0
current_holdings = pd.DataFrame(columns=['Stock', 'Buy_Price', 'Shares_Owned', 'Buy_Date'])
trade_log = []
history = []

print(f"Running backtest from {dates[0].date()} to {dates[-1].date()}...")


for date in dates:
    day_data = data[data['Date'] == date]
    
    picks = get_predictions(date, binary_model, cont_model)
    target_tickers = picks.sort_values('val', ascending=False).head(10)['ticker'].tolist() if not picks.empty else []


    if not current_holdings.empty:

        prices = day_data[['Ticker', 'Close']].rename(columns={'Ticker': 'Stock'})
        current_holdings = current_holdings.merge(prices, on='Stock', how='left')
        

        keep_mask = current_holdings['Stock'].isin(target_tickers)
        to_sell = current_holdings[~keep_mask]
        
        for _, row in to_sell.iterrows():
            sell_price = row['Close'] if pd.notnull(row['Close']) else row['Buy_Price']
            proceeds = sell_price * row['Shares_Owned']
            cash += proceeds
            trade_log.append({'Date': date, 'Ticker': row['Stock'], 'Action': 'SELL', 'Price': sell_price})
            
        current_holdings = current_holdings[keep_mask].drop(columns=['Close'])
    for ticker in target_tickers:
        if len(current_holdings) >= 10: break
        if ticker in current_holdings['Stock'].values: continue
        if cash < per_position: break
            
        price_row = day_data[day_data['Ticker'] == ticker]
        if not price_row.empty:
            buy_price = price_row['Close'].iloc[0]
            shares = per_position / buy_price
            
            new_row = pd.DataFrame([{
                'Stock': ticker, 'Buy_Price': buy_price, 
                'Shares_Owned': shares, 'Buy_Date': date
            }])
            current_holdings = pd.concat([current_holdings, new_row], ignore_index=True)
            cash -= per_position
            trade_log.append({'Date': date, 'Ticker': ticker, 'Action': 'BUY', 'Price': buy_price})


    if not current_holdings.empty:
        daily_prices = day_data.set_index('Ticker')['Close']

        stock_value = (current_holdings['Shares_Owned'] * current_holdings['Stock'].map(daily_prices).fillna(current_holdings['Buy_Price'])).sum()
    else:
        stock_value = 0.0
    
    total_val = stock_value + cash
    history.append({'Date': date, 'Total_Value': total_val})
    print(f"Date: {date.date()} | Holdings: {len(current_holdings)} | Value: ${total_val:,.2f}", end='\r')


history_df = pd.DataFrame(history)
final_value = history_df['Total_Value'].iloc[-1]
total_return = ((final_value - initial_capital) / initial_capital) * 100


print(f"FINAL PORTFOLIO VALUE: ${final_value:,.2f}")
print(f"TOTAL RETURN:         {total_return:.2f}%")
print(f"TOTAL TRADES:         {len(trade_log)}")



history_df = history_df[history_df['Date'] <= '2025-12-19']


sp500 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/SP500testdata.csv")
shares = float(initial_capital / sp500['Open'][0])

sp500['Shares'] = shares
sp500['Value'] = sp500['Shares'] * sp500['Close']
sp500['Date'] = pd.to_datetime(sp500['Date'])
sp500['initial_val'] = sp500['Open'][0] * shares

initial_sp500 = float(sp500['Value'][0])

sp500_length = len(sp500) -1
final_sp500 = sp500['Value'][sp500_length]
sp500_return = float((final_sp500 - initial_sp500) / initial_sp500 )*100

model_return_diff = total_return - sp500_return


plt.figure(figsize=(12, 6))
plt.plot(history_df['Date'], history_df['Total_Value'], color="#130d7a", linewidth=2, label='Strategy Equity Curve')
plt.plot(sp500['Date'], sp500['Value'], color = '#2ca02c', linewidth=2, label='SP500 Curve')
plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.5, label='Starting Capital')
plt.title('Backtest Results: Portfolio Value from Nov 12 2025')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.xticks(rotation=45) # Slant by 45 degrees
plt.show()


trade_log_df = pd.DataFrame(trade_log).to_csv('trade_log.csv', index=False)

#Total Return = 9.07%
#Sp500 return = .553%
