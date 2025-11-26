import pandas as pd
data = pd.read_csv("/Users/lukeromes/Desktop/Notre Dame/Mod2/Machine Learning/Data/combined_data.csv",header=None)
data = data[~data[0].isin(['Ticker', 'Date'])]
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
data = data.reset_index(drop=True)
data = data.iloc[1:]





data.to_csv("finalsp500.csv", index= False)






