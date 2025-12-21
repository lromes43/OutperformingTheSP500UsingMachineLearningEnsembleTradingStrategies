def binary_prediction_func(data, prediction_date):
    import pandas as pd
    import joblib
    import xgboost as xgb

    binary = joblib.load(
        "/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/FinalBoostedFiveDayClassifier.job.lib"
    )

    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    X = pd.get_dummies(df, columns=['Ticker'], drop_first=True)

    X = X.drop(columns=[
        'Movement',
        'next_day_pct_change',
        'Movement_5_day',
        'next_5_day_pct_change',
        'Movement_30_day',
        'next_30_day_pct_change'
    ], errors='ignore')

    X['Date'] = X['Date'].apply(lambda x: x.timestamp())
    X = X.reindex(columns=binary.feature_names, fill_value=0)

    dmatrix = xgb.DMatrix(X)
    preds = binary.predict(dmatrix)

    results = pd.DataFrame({
        'Date': df['Date'].dt.date,
        'Ticker': df['Ticker'],
        'Predicted_Movement': preds
    })

    results['Date'] = pd.to_datetime(results['Date'])
    target_date = pd.to_datetime(prediction_date)

    results = results[results['Date'].dt.date == target_date.date()]

    threshold = results['Predicted_Movement'].quantile(0.75)
    results['Buy'] = (results['Predicted_Movement'] >= threshold).astype(int)

    return results[['Date', 'Ticker', 'Predicted_Movement', 'Buy']]


def cont_prediction_func(data, prediction_date):
    import pandas as pd
    import joblib
    import xgboost as xgb

    cont = joblib.load(
        "/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/ContinuousFiveDayFinal.joblib"
    )

    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    X = df.drop(columns=[
        'Daily_Return',
        'next_day_pct_change',
        'next_5_day_pct_change',
        'Movement_5_day',
        'next_30_day_pct_change',
        'Movement_30_day',
        'Movement'
    ], errors='ignore')

    identifiers = X[['Date', 'Ticker']].copy()

    X = pd.get_dummies(X, columns=['Ticker'], drop_first=True)
    X['Date'] = X['Date'].apply(lambda x: x.timestamp())
    X = X.reindex(columns=cont.feature_names, fill_value=0)

    dpred = xgb.DMatrix(X)
    preds = cont.predict(dpred)

    results = identifiers.copy() 
    results['Predicted_5_Day_Pct_Change'] = preds
    results['Date'] = pd.to_datetime(results['Date'])

    target_date = pd.to_datetime(prediction_date)
    results = results[results['Date'].dt.date == target_date.date()]

    threshold = results['Predicted_5_Day_Pct_Change'].quantile(0.75)
    results['Buy'] = (results['Predicted_5_Day_Pct_Change'] >= threshold).astype(int)

    return results


def model_results_merging(binary_df, cont_df):
    import pandas as pd

    binary_df['Date'] = pd.to_datetime(binary_df['Date']).dt.tz_localize(None)
    cont_df['Date'] = pd.to_datetime(cont_df['Date']).dt.tz_localize(None)

    merged = pd.merge(
        binary_df,
        cont_df,
        on=['Date', 'Ticker'],
        how='inner',
        suffixes=('_binary', '_cont')
    )

    merged['Buy'] = ((merged['Buy_binary'] == 1) & (merged['Buy_cont'] == 1)).astype(int)
    merged = merged[merged['Buy'] == 1] 



    merged = merged.sort_values(
        'Predicted_Pct_Change',
        ascending=False
    ).head(10)

    final_cols = [
        'Date',
        'Ticker',
        'Predicted_Movement',          
        'Predicted_Pct_Change', 
        'Buy'
    ]

    merged = merged[final_cols]
    merged = merged.rename(columns={'Predicted_Pct_Change': 'Predicted_5_Day_Change'})


    output_path = (
        "/Users/lukeromes/Desktop/Personal/Sp500Project/"
        "DailyPredictions/FiveDay/Five_Day_Merged_Final.csv"
    )
    merged.to_csv(output_path, index=False)
    print(f"Saved clean output to {output_path}")

    

    return merged
