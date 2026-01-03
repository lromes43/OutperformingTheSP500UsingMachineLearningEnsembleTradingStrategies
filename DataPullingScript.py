import sys
sys.path.append('/Users/lukeromes/Desktop/Personal/Sp500Project/Functions')
import sp500_pipeline
import split
import pandas as pd
from datetime import date

date1_in_range = "2024-11-11"
date2_in_range = "2025-11-12"

data_path = "/Users/lukeromes/Desktop/Personal/Sp500Project/Data/FINALSP500Data.feather"
train_start_date = pd.to_datetime(date1_in_range).tz_localize(None)
train_end_date = pd.to_datetime("2025-10-08").tz_localize(None)
test_start_date = pd.to_datetime("2025-10-09").tz_localize(None)



sp500_pipeline.run_sp500_pipeline(date1_in_range , date2_in_range )
split.train_test_split_by_date_function(data_path, train_start_date, train_end_date, test_start_date)


data = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/FINALSP500Data.feather")








