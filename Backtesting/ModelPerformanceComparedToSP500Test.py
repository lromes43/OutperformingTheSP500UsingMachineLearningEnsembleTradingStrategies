import pandas as pd
import joblib
import xgboost as xgb


data = pd.read_feather("/Users/lukeromes/Desktop/Sp500Project/Data/TestData.feather")
dates = sorted(data['Date'].unique())
first_date = dates[0]

cash = 0
per_position_security = 1000/10

removed_holdings = pd.DataFrame(columns=['Stock','Sell_Date','Sell_Price','Shares_Owned','Proceeds'])
trade_log = []
daily_portfolio = pd.DataFrame(columns=['Date','Holdings_Value','Cash','Total_Value'])


def binary_preprocessing(date, file_path, predictor):
    binary_model = joblib.load("/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/NEWONEDAYBOOSTED.job.lib")
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
    pred = binary_model.predict(dtest)
    pred_final = (pred >= 0.5).astype(int)
    pred_final_df = pd.DataFrame(pred_final).reset_index().rename(columns={'index':'iteration', 0:'predicted_up'})
    actual_df = actual.reset_index().reset_index().drop('index', axis=1).rename(columns={'level_0':'iteration', predictor:'actual_up'})
    merged_binary = pd.merge(pred_final_df, actual_df, on='iteration')
    merged_binary['ticker'] = subset['Ticker'].sort_values().reset_index(drop=True)
    merged_binary['buy'] = merged_binary['actual_up'].apply(lambda x: 1 if x == 1 else 0)
    return merged_binary[merged_binary['buy'] == 1].copy()


def cont_preprocessing(date, file_path, predictor):
    cont_model = joblib.load("/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/ONEDAYCONTNEW.joblib")
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
    X_final = X_raw.reindex(columns=cont_model.feature_names, fill_value=0).fillna(0)
    dtest = xgb.DMatrix(X_final)
    y_pred = cont_model.predict(dtest)
    merged = pd.DataFrame({
        'iteration': range(len(y_pred)),
        'Predicted_Pct_Chg': y_pred,
        'ticker': subset['Ticker'].sort_values().reset_index(drop=True)
    })
    merged['buy'] = merged['Predicted_Pct_Chg'].apply(lambda x: 1 if x > 0.00 else 0)
    return merged[merged['buy'] == 1].copy()

Day_1_Holdings = []
Value = []

column_names = ['Ticker', 'Date', 'Buy_Val', 'Sell_Val', 'Shares']

sells_df = pd.DataFrame(columns=column_names)


for i in data['Date']: 
    if i == 0: 
        binary = binary_preprocessing(i, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "Movement")
        cont = cont_preprocessing(i, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "next_day_pct_change")
        merged = pd.merge(binary, cont, on=['iteration','ticker','buy'])
        merged = merged.sort_values("Predicted_Pct_Chg", ascending=False)
        daily = merged.head(10)
        daily_prices_df = pd.DataFrame(data['Ticker'], data['Close'], data['Date'])
        Day_1_Merged = pd.merge(daily, daily_prices_df, how = 'inner', on = ['Ticker', 'Date'])
        Day_1_Merged['Shares'] = Day_1_Merged['Close'] / per_position_security
        Day_1_Merged['Value'] = Day_1_Merged['Close'] * Day_1_Merged['Shares']
        Day_1_Value = Day_1_Merged['Value'].sum()
        Value.append(Day_1_Value)
    if i ==1: 
        binary = binary_preprocessing(i, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "Movement")
        cont = cont_preprocessing(i, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "next_day_pct_change")
        merged = pd.merge(binary, cont, on=['iteration','ticker','buy'])
        merged = merged.sort_values("Predicted_Pct_Chg", ascending=False)
        daily = merged.head(10)
        daily_prices_df = pd.DataFrame(data['Ticker'], data['Close'], data['Date'])
        Daily_Merged = pd.merge(daily, daily_prices_df, how = 'inner', on = ['Ticker', 'Date'])
        Daily_Merged['Value'] =  Daily_Merged['Close'] *  Daily_Merged['Shares']
        Value = Daily_Merged['Value'].sum()
        Value.append(Value)
        holdings = Day_1_Merged.copy()
        for ticker in Daily_Merged['ticker']: 
            if ticker not in holdings['Ticker']:
                holdings = holdings.drop(holdings[holdings['Ticker']=='ticker'])
                sells_df.loc[len(sells_df)] = ["ticker", data['Date'][i], holdings['Close'],Daily_Merged['Close'], holdings['Shares']]
                cash += sells_df.loc[len(sells_df)]['Shares'] * sells_df.loc[len(sells_df)]['Close']
            else: 
                continue
    if i > 1: 
        binary = binary_preprocessing(i, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "Movement")
        cont = cont_preprocessing(i, "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/ModelFuturePerformanceDataCleaned.feather", "next_day_pct_change")
        merged = pd.merge(binary, cont, on=['iteration','ticker','buy'])
        merged = merged.sort_values("Predicted_Pct_Chg", ascending=False)
        daily = merged.head(10)
        daily_prices_df = pd.DataFrame(data['Ticker'], data['Close'], data['Date'])
        Daily_Merged = pd.merge(daily, daily_prices_df, how = 'inner', on = ['Ticker', 'Date'])
        Daily_Merged['Value'] =  Daily_Merged['Close'] *  Daily_Merged['Shares']
        Daily_Merged_2 = Daily_Merged
        Value = Daily_Merged_2['Value'].sum()
        Value.append(Value)
        holdings = Daily_Merged.copy()
        for ticker in Daily_Merged_2['ticker']: 
            if ticker not in holdings['Ticker']:
                holdings = holdings.drop(holdings[holdings['Ticker']=='ticker'])
                sells_df.loc[len(sells_df)] = ["ticker", data['Date'][i], holdings['Close'],Daily_Merged_2['Close'], holdings['Shares']]
                cash += sells_df.loc[len(sells_df)]['Shares'] * sells_df.loc[len(sells_df)]['Close']
            else: 
                continue






total_profit = Value - (per_position_security * 10)
total_return_pct = (total_profit / per_position_security * 10) * 100
        

                



           



        











print("\n--- FINAL SUMMARY ---")
print(f"Initial capital invested: ${initial_capital:,.2f}")
print(f"Final portfolio value:   ${portfolio_value:,.2f}")
print(f"Total profit:            ${total_profit:,.2f}")
print(f"Total return:            {total_return_pct:.2f}%")
print(f"Number of sells recorded: {len(removed_holdings)}")
print(f"Number of buys recorded:  {len(trade_log)}")
