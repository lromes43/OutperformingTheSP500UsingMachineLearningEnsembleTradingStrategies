import sys
sys.path.append('/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2/Functions')
import sp500_pipeline
import split
import pandas as pd
from datetime import date

date1_in_range = "2025-12-10"
date2_in_range = "2025-12-18"
data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2/FinalTestData.feather")
prediction_date = '2025-12-17'
data1 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2/DailyPredictions/Results_df_filtered_binary.csv")
data2 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2/DailyPredictions/Results_df_filtered_cont.csv")
data_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/RetrainingModel2/Data/FinalTestData.feather"
train_end_date = "2025-11-9"
test_start_date = "2025-11-10"



sp500_pipeline.run_sp500_pipeline(date1_in_range , date2_in_range )

split.train_test_split_by_date_function(data_path, train_end_date, test_start_date)










