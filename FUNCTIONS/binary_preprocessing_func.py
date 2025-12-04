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