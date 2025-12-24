import sys
sys.path.append('/Users/lukeromes/Desktop/Personal/Sp500Project/Functions')
import sp500_pipeline
import split
import pandas as pd
from datetime import date

date1_in_range = "2024-12-1"
date2_in_range = "2025-12-19"
data1 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Results_df_filtered_binary.csv")
data2 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/DailyPredictions/Results_df_filtered_cont.csv")
data_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/FinalTestData.feather"
train_end_date = pd.to_datetime("2025-11-09").tz_localize(None)
test_start_date = pd.to_datetime("2025-11-10").tz_localize(None)



un_sp500_pipeline.run_sp500_pipeline(date1_in_range , date2_in_range )
split.train_test_split_by_date_function(data_path, train_end_date, test_start_date)







