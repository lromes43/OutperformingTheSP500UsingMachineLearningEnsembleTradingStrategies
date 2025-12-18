
#Fitting Binary Model


def binary_prediction_func(data, prediction_date):
    import pandas as pd
    import pandas as pd
    import joblib
    import xgboost as xgb

    binary = joblib.load("/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/FinalBoostedOneDayClassifier.job.lib")
    model_feature_names = binary.feature_names 

    df = data.copy() 
    df['Date'] = pd.to_datetime(df['Date'], utc=True) 

    features_for_ohe = pd.get_dummies(df, columns=['Ticker'], prefix='Ticker', drop_first=True)

    columns_to_drop = [
        'Movement',       
        'next_day_pct_change',    
        'Movement_5_day',     
        'next_5_day_pct_change',  
        'Movement_30_day',    
        'next_30_day_pct_change', 
    ]

    X_train_processed = features_for_ohe.drop(columns=columns_to_drop, errors='ignore')

    X_train_final = X_train_processed.copy()

    X_train_final['Date'] = X_train_final['Date'].apply(lambda x: x.timestamp())

    X_train_aligned = X_train_final.reindex(columns=model_feature_names, fill_value=0)

    y_train_final = df['Movement'].astype(int)
    dtrain = xgb.DMatrix(X_train_aligned, label=y_train_final)


    predictions1 = binary.predict(dtrain)


    results_df = pd.DataFrame({
        'Date': df['Date'].dt.date, 
        'Ticker': df['Ticker'],     
        'Predicted_Movement': predictions1 
    })

    print(results_df.head(10)) 

    results_df['Date'] = pd.to_datetime(results_df['Date'])

    target_date = pd.to_datetime(prediction_date)


    results_df_filtered = results_df[results_df['Date'].dt.date == target_date.date()]

    thirdquartile_movement = float(results_df.describe(include='all').iloc[8,2])
    secondquartile = float(results_df.describe(include='all').iloc[7,2])

    threshold = (thirdquartile_movement + secondquartile)/2

    results_df_filtered['Buy'] = (results_df_filtered['Predicted_Movement'] >= threshold).astype(int)
    results_df_filtered_binary = results_df_filtered
    results_df_filtered_binary = results_df_filtered_binary.sort_values('Predicted_Movement', ascending=False)
    results_df_filtered_binary.to_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Results_df_filtered_binary.csv")
    return results_df_filtered_binary

import pandas as pd
binary_prediction_func(data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/FinalTestData.feather"), prediction_date = '2025-12-16')


#Fitting cont model 

def cont_prediction_func(data, prediction_date):
    import pandas as pd
    import joblib
    import xgboost as xgb
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    cont = joblib.load("/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/ContinuousOneDayFinal.joblib")

    try:
        model_feature_names = cont.feature_names
    except AttributeError:

        print("Warning: Feature names not found on 'cont' object. Using placeholder.")

        pass

    X_pred_raw = data.copy()


    X_pred_raw['Date'] = pd.to_datetime(X_pred_raw['Date'], utc=True) 


    columns_to_drop = [
        'Daily_Return',
        'next_day_pct_change',
        'next_5_day_pct_change',
        'Movement_5_day',
        'next_30_day_pct_change',
        'Movement_30_day',
        'Movement',
    
    ]


    X_pred_processed = X_pred_raw.drop(columns=columns_to_drop, errors='ignore')


    identifiers_df = X_pred_processed[['Date', 'Ticker']].copy()


    X_pred_ohe = pd.get_dummies(X_pred_processed, columns=['Ticker'], prefix='Ticker', drop_first=True)


    if 'Date' in X_pred_ohe.columns:
        X_pred_ohe['Date'] = X_pred_ohe['Date'].apply(lambda x: x.timestamp())
        X_pred_final = X_pred_ohe.apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        raise KeyError("The 'Date' column is missing from prediction features. Please check data loading.")

    X_pred_aligned = X_pred_final.reindex(columns=model_feature_names, fill_value=0)


    dpred = xgb.DMatrix(X_pred_aligned, feature_names=model_feature_names)

    predictions = cont.predict(dpred)


    results_df = identifiers_df.copy()
    results_df['Predicted_Pct_Change'] = predictions


    import pandas as pd
    import numpy as np 

    results_df['Date'] = pd.to_datetime(results_df['Date'])

    target_date = pd.to_datetime(prediction_date)


    results_df_filtered_cont = results_df[results_df['Date'].dt.date == target_date.date()]


    thirdquartile_movement = float(results_df_filtered_cont .describe(include='all').iloc[8,2])
    secondquartile = float(results_df_filtered_cont .describe(include='all').iloc[7,2])

    threshold = (thirdquartile_movement + secondquartile)/2

    results_df_filtered_cont ['Buy'] = (results_df_filtered_cont ['Predicted_Pct_Change'] >= threshold).astype(int)
    results_df_filtered_cont = results_df_filtered_cont 

    results_df_filtered_cont = results_df_filtered_cont.sort_values('Predicted_Pct_Change', ascending=False)
    results_df_filtered_cont.to_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Results_df_filtered_cont.csv")
    return results_df_filtered_cont

import pandas as pd
cont_prediction_func(data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/FinalTestData.feather"), prediction_date = '2025-12-16')



def model_results_merging(data1, data2):
    import pandas as pd
    data1['Date'] = pd.to_datetime(data1['Date'])
    data2['Date'] = pd.to_datetime(data2['Date'])

    if data1['Date'].dt.tz is not None:
        data1['Date'] = data1['Date'].dt.tz_localize(None)
    if data2['Date'].dt.tz is not None:
        data2['Date'] = data2['Date'].dt.tz_localize(None)

    print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}")

    merged = pd.merge(
        data2, 
        data1, 
        how='inner', 
        on=['Date', 'Ticker', 'Buy']
    )

    if merged.empty:
        print("Warning: Merged dataframe is empty! Check if Date/Ticker/Buy values actually match.")
        return merged

    
    merged_final = merged[merged['Buy'] == 1].copy()
    merged_final = merged_final.sort_values('Predicted_Pct_Change', ascending=False)
    merged_final = merged_final.head(10)

    output_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Merged_Final.csv"
    merged_final.to_csv(output_path, index=False)
    print(f"Successfully saved to {output_path}")
    
    return merged_final

import pandas as pd
data1 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Results_df_filtered_binary.csv")
data2 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Results_df_filtered_cont.csv")

result = model_results_merging(data1, data2)