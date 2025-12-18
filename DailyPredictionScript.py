import sys
sys.path.append('/Users/lukeromes/Desktop/Personal/Sp500Project/Functions')
import sp500_pipeline
import dailymodels
import pandas as pd

date1_in_range = "2025-12-10"
date2_in_range = "2025-12-18"
data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/FinalTestData.feather")
prediction_date = '2025-12-17'
data1 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Results_df_filtered_binary.csv")
data2 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Results_df_filtered_cont.csv")




sp500_pipeline.run_sp500_pipeline(date1_in_range , date2_in_range )



dailymodels.binary_prediction_func(data, prediction_date)

dailymodels.cont_prediction_func(data, prediction_date)

dailymodels.model_results_merging(data1, data2)








