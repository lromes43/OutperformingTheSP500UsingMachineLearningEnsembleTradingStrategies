import pandas as pd
import joblib
import xgboost as xgb


data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather")


per_position_security = 10000


cash = 0.0
removed_holdings = pd.DataFrame(columns=['Stock','Sell_Date','Sell_Price','Shares_Owned','Proceeds'])
trade_log = []  


def binary_preprocessing(date, file_path, predictor):
    binaryone = joblib.load("/Users/lukeromes/Desktop/Personal/Sp500Project/Models/FinalBoostedOneDayClassifier.joblib")
    df = pd.read_feather(file_path)
    subset = df[df['Date'] == date]
    drop_cols = [
        'Date', 'next_day_pct_change','Daily_Return',
        'next_5_day_pct_change','Movement_5_day',
        'next_30_day_pct_change','Movement_30_day',
        'Movement'
    ]
    X = subset.drop(drop_cols, axis=1)
    X_final = pd.get_dummies(X, drop_first=True)
    actual = subset[predictor].astype(int)
    dtest = xgb.DMatrix(X_final, label=actual)
    pred = binaryone.predict(dtest)
    pred_final = (pred >= 0.5).astype(int)
    pred_final_df = pd.DataFrame(pred_final).reset_index().rename(columns={'index':'iteration', 0:'predicted_up'})
    actual_df = actual.reset_index().reset_index().drop('index', axis=1).rename(columns={'level_0':'iteration', predictor:'actual_up'})
    merged_binary = pd.merge(pred_final_df, actual_df, on='iteration')
    merged_binary['ticker'] = subset['Ticker'].sort_values().reset_index(drop=True)
    merged_binary['buy'] = merged_binary['actual_up'].apply(lambda x: 1 if x == 1 else 0)
    return merged_binary[merged_binary['buy'] == 1].copy()



def cont_preprocessing(date, file_path, predictor):
    continuousone = joblib.load("/Users/lukeromes/Desktop/Personal/Sp500Project/Models/ContinuousOneDayFinal.joblib")
    df = pd.read_feather(file_path)
    subset = df[df['Date'] == date]
    drop_cols = [
        'Date','next_day_pct_change','Daily_Return',
        'next_5_day_pct_change','Movement_5_day',
        'next_30_day_pct_change','Movement_30_day',
        'Movement'
    ]
    X = subset.drop(drop_cols, axis=1)
    X_raw = pd.get_dummies(X, drop_first=True)
    X_final = X_raw.reindex(columns=continuousone.feature_names, fill_value=0).fillna(0)
    dtest = xgb.DMatrix(X_final)
    y_pred = continuousone.predict(dtest)
    merged = pd.DataFrame({
        'iteration': range(len(y_pred)),
        'Initial_Predicted': y_pred * 100,
        'ticker': subset['Ticker'].sort_values().reset_index(drop=True)
    })
    merged['buy'] = merged['Initial_Predicted'].apply(lambda x: 1 if x > 0 else 0)
    return merged[merged['buy'] == 1].copy()



dates = sorted(data['Date'].unique())
first_date = dates[0]

try:
    day1holdingsmerge

    if day1holdingsmerge.empty:
        raise NameError
except Exception:
    b1 = binary_preprocessing(first_date, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "Movement")
    c1 = cont_preprocessing(first_date, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "next_day_pct_change")
    d1 = pd.merge(b1, c1, on=['iteration','ticker','buy']).sort_values("Initial_Predicted", ascending=False).head(10)

    prices_day1 = data[data['Date'] == first_date][['Ticker','Close']]
    day1holdingsmerge = pd.merge(
        d1.rename(columns={'ticker':'Stock'}),
        prices_day1,
        how='left',
        left_on='Stock',
        right_on='Ticker'
    ).rename(columns={'Close':'Buy_Price'})


    day1holdingsmerge['Shares_Owned'] = (per_position_security / day1holdingsmerge['Buy_Price']).fillna(0)
    day1holdingsmerge['Buy_Date'] = first_date
    day1holdingsmerge['current_price'] = day1holdingsmerge['Buy_Price']
    day1holdingsmerge = day1holdingsmerge[['Stock','Buy_Price','Shares_Owned','Buy_Date','current_price']]

initial_capital = len(day1holdingsmerge) * per_position_security


for date in dates:


    binary = binary_preprocessing(date, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "Movement")
    cont = cont_preprocessing(date, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "next_day_pct_change")
    merged = pd.merge(binary, cont, on=['iteration','ticker','buy'])
    merged = merged.sort_values("Initial_Predicted", ascending=False)
    daily = merged.head(10)


    if not day1holdingsmerge.empty:
        sell_mask = ~day1holdingsmerge['Stock'].isin(daily['ticker'].values)
        to_sell = day1holdingsmerge.loc[sell_mask].copy()
        if not to_sell.empty:
            to_sell['Proceeds'] = to_sell['current_price'] * to_sell['Shares_Owned']
            sell_value = to_sell['Proceeds'].sum()
            cash += sell_value

            to_sell['Sell_Date'] = date
            to_sell = to_sell[['Stock','Sell_Date','current_price','Shares_Owned','Proceeds']].rename(columns={'current_price':'Sell_Price'})
            removed_holdings = pd.concat([removed_holdings, to_sell], ignore_index=True)
            day1holdingsmerge = day1holdingsmerge.loc[~sell_mask].reset_index(drop=True)

    if cash > 0 and not daily.empty:
        candidates = [t for t in daily['ticker'].tolist() if t not in day1holdingsmerge['Stock'].values]
        if len(candidates) > 0:
            buy_ticker = candidates[0]
            price_row = data[(data['Date'] == date) & (data['Ticker'] == buy_ticker)]
            if not price_row.empty:
                price = price_row['Close'].iloc[0]
                shares_to_buy = cash / price
                new_row = {
                    'Stock': buy_ticker,
                    'Buy_Price': price,
                    'Shares_Owned': shares_to_buy,
                    'Buy_Date': date,
                    'current_price': price
                }
                day1holdingsmerge = pd.concat([day1holdingsmerge, pd.DataFrame([new_row])], ignore_index=True)
                trade_log.append({'Date':date,'Ticker':buy_ticker,'Action':'BUY','Price':price,'Shares':shares_to_buy,'Cash_before':cash})
                cash = 0.0

    if not day1holdingsmerge.empty:
        day_prices = data[data['Date'] == date][['Ticker','Close']].rename(columns={'Ticker':'Stock','Close':'current_price'})
        day1holdingsmerge = pd.merge(
            day1holdingsmerge.drop(columns=['current_price'], errors='ignore'),
            day_prices,
            how='left',
            on='Stock'
        )
        day1holdingsmerge['current_price'] = day1holdingsmerge['current_price'].fillna(day1holdingsmerge['Buy_Price'])

    holdings_value = 0.0 if day1holdingsmerge.empty else (day1holdingsmerge['current_price'] * day1holdingsmerge['Shares_Owned']).sum()
    total_value = holdings_value + cash
    print(f"{date}  Holdings: {len(day1holdingsmerge)}  Cash: ${cash:,.2f}  Portfolio Value: ${total_value:,.2f}")


holdings_value = 0.0 if day1holdingsmerge.empty else (day1holdingsmerge['current_price'] * day1holdingsmerge['Shares_Owned']).sum()
portfolio_value = holdings_value + cash

total_profit = portfolio_value - initial_capital
total_return_pct = (total_profit / initial_capital) * 100

print("\n--- FINAL SUMMARY ---")
print(f"Initial capital invested: ${initial_capital:,.2f}")
print(f"Final portfolio value:   ${portfolio_value:,.2f}")
print(f"Total profit:            ${total_profit:,.2f}")
print(f"Total return:            {total_return_pct:.2f}%")
print(f"Number of sells recorded: {len(removed_holdings)}")
print(f"Number of buys recorded:  {len(trade_log)}")



removed_holdings.to_csv('/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/removed_holdings.csv', index=False)

trade_log_df = pd.DataFrame(trade_log)
trade_log_df.to_csv('/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/trade_log_df.csv', index = False)

#Comparing Model to SP500

sp500table = pd.read_html("https://finance.yahoo.com/quote/%5EGSPC/history/")[1] #getting to mahy request did i break the API lol
SP500_startprice = 6415.54
sp500_endprice = 6486.61

SP500_capital = 100000
SP500_shares = SP500_capital / SP500_startprice
SP_500_initial = SP500_shares * SP500_startprice
SP_500_finish = SP500_shares * sp500_endprice
SP500_profit = SP_500_finish - SP500_capital 
SP_500_total_return_pct = (SP500_profit/SP500_capital) * 100
