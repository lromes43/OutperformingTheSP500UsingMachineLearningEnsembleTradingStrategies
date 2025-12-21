import sys
import pandas as pd
sys.path.append('/Users/lukeromes/Desktop/Personal/Sp500Project/Functions')
import sp500_pipeline
import FiveDayModels


date1_in_range = "2025-12-8"
date2_in_range = "2025-12-15"
sp500_pipeline.run_sp500_pipeline(start_date = date1_in_range , end_date = date2_in_range)


data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/FinalTestData.feather")
prediction_date = '2025-12-12'



FiveDayModels.binary_prediction_func(data, prediction_date)

FiveDayModels.cont_prediction_func(data, prediction_date)


data1 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/FiveDay/Five_Day_Results_df_filtered_binary.csv")
data2 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/FiveDay/Five_Day_Results_df_filtered_cont.csv")



FiveDayModels.model_results_merging(data1, data2)








