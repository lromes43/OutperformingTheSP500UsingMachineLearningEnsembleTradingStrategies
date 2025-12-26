def binary_prediction_func(data, prediction_date):
    import pandas as pd
    import joblib
    import xgboost as xgb

    # Load model
    binary = joblib.load(
        "/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/FinalBoostedOneDayClassifier.job.lib"
    )
    model_feature_names = binary.feature_names

    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Preserve identifiers
    identifiers = df[['Date', 'Ticker']].copy()

    # One-hot encode
    X = pd.get_dummies(df, columns=['Ticker'], prefix='Ticker', drop_first=True)

    # Drop targets
    columns_to_drop = [
        'Movement',
        'next_day_pct_change',
        'Movement_5_day',
        'next_5_day_pct_change',
        'Movement_30_day',
        'next_30_day_pct_change',
    ]
    X = X.drop(columns=columns_to_drop, errors='ignore')

    # Convert date to timestamp
    X['Date'] = X['Date'].apply(lambda x: x.timestamp())

    # Align to model features
    X = X.reindex(columns=model_feature_names, fill_value=0)

    # Predict
    dpred = xgb.DMatrix(X)
    predictions = binary.predict(dpred)

    # Results
    results_df = identifiers.copy()
    results_df['Predicted_Movement'] = predictions
    results_df['Date'] = pd.to_datetime(results_df['Date'])

    # Filter prediction date
    target_date = pd.to_datetime(prediction_date)
    results_df = results_df[
        results_df['Date'].dt.date == target_date.date()
    ].copy()

    # Threshold via quantiles
    q3 = results_df['Predicted_Movement'].quantile(0.75)
    q2 = results_df['Predicted_Movement'].quantile(0.50)
    threshold = (q3 + q2) / 2

    results_df.loc[:, 'Buy'] = (
        results_df['Predicted_Movement'] >= threshold
    ).astype(int)

    results_df = results_df.sort_values(
        'Predicted_Movement', ascending=False
    )

    output_path = (
        "/Users/lukeromes/Desktop/Personal/Sp500Project/"
        "DailyPredictions/OneDay/Results_df_filtered_binary.csv"
    )
    results_df.to_csv(output_path, index=False)

    return results_df



def cont_prediction_func(data, prediction_date):
    import pandas as pd
    import joblib
    import xgboost as xgb
    import warnings

    warnings.filterwarnings("ignore")

    # Load trained continuous regression model
    model = joblib.load(
        "/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/ContinuousOneDayFinal.joblib"
    )
    model_feature_names = model.feature_names

    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Keep identifiers for output
    identifiers = df[['Date', 'Ticker']].copy()

    # Drop ALL target / leakage columns
    target_cols = [
        'next_day_pct_change',
        'Movement',
        'next_5_day_pct_change',
        'Movement_5_day',
        'next_30_day_pct_change',
        'Movement_30_day'
    ]

    X = df.drop(columns=target_cols, errors='ignore')

    # One-hot encode ticker
    X = pd.get_dummies(X, columns=['Ticker'], prefix='Ticker', drop_first=True)

    # Convert date to numeric timestamp
    X['Date'] = X['Date'].astype('int64') // 10**9

    # Ensure numeric safety
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Align with training features
    X = X.reindex(columns=model_feature_names, fill_value=0)

    # Predict
    dmatrix = xgb.DMatrix(X)
    predictions = model.predict(dmatrix)

    # Build results dataframe
    results = identifiers.copy()
    results['Predicted_Next_Day_Pct_Change'] = predictions
    results['Date'] = pd.to_datetime(results['Date'])

    # Filter to prediction date
    target_date = pd.to_datetime(prediction_date).date()
    results = results[
        results['Date'].dt.date == target_date
    ].copy()

    # Optional: Buy signal (top 25%)
    q75 = results['Predicted_Next_Day_Pct_Change'].quantile(0.75)
    results['Buy'] = (
        results['Predicted_Next_Day_Pct_Change'] >= q75
    ).astype(int)

    # Sort by expected return
    results = results.sort_values(
        'Predicted_Next_Day_Pct_Change',
        ascending=False
    )

    # Save output
    output_path = (
        "/Users/lukeromes/Desktop/Personal/Sp500Project/"
        "DailyPredictions/OneDay/Results_df_filtered_cont.csv"
    )
    results.to_csv(output_path, index=False)

    return results


def model_results_merging(binary_df, cont_df):
    import pandas as pd

    # Clean columns
    binary_df = binary_df.loc[:, ~binary_df.columns.str.contains('^Unnamed')]
    cont_df = cont_df.loc[:, ~cont_df.columns.str.contains('^Unnamed')]

    binary_df['Date'] = pd.to_datetime(binary_df['Date'])
    cont_df['Date'] = pd.to_datetime(cont_df['Date'])

    # Merge on Date, Ticker, Buy
    merged = pd.merge(
        cont_df,
        binary_df[['Date', 'Ticker', 'Buy']],
        how='inner',
        on=['Date', 'Ticker', 'Buy']
    )

    if merged.empty:
        print("Warning: merged dataframe is empty")
        return merged

    merged_final = (
        merged[merged['Buy'] == 1]
        .sort_values('Predicted_Next_Day_Pct_Change', ascending=False)
        .head(10)
    )

    output_path = (
        "/Users/lukeromes/Desktop/Personal/Sp500Project/"
        "DailyPredictions/OneDay/One_Day_Merged_Final.csv"
    )
    merged_final.to_csv(output_path, index=False)

    return merged_final
