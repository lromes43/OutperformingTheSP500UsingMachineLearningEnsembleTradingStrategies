import sys
import pandas as pd
sys.path.append('/Users/lukeromes/Desktop/Personal/Sp500Project/Functions')
import russel2000_pipeline
import Russell2000DailyModels


date1_in_range = "2025-12-10"
date2_in_range = "2025-12-23"
russel2000_pipeline.micro_pipeline(start_date = date1_in_range , end_date = date2_in_range)


data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/FinalTestData.feather")
prediction_date = '2025-12-22'



Russell2000DailyModels.binary_prediction_func(data, prediction_date)

Russell2000DailyModels.cont_prediction_func(data, prediction_date)


data1 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/OneDay/Results_df_filtered_cont.csv")
data2 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/OneDay/Results_df_filtered_cont.csv")



Russell2000DailyModels.model_results_merging(data1, data2)


