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