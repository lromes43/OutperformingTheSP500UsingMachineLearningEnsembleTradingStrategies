
import os
import pandas as pd

# Path to the folder containing all CSV files
folder_path = "/Users/lukeromes/Desktop/Notre Dame/Mod2/Machine Learning/sp500_yearly_data"


# List to hold all dataframes
all_data = []

# Loop over each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV
        df = pd.read_csv(file_path)
        
        # Add filename as a new column (without extension)
        df['source_file'] = os.path.splitext(file_name)[0]
        
        # Append to list
        all_data.append(df)

# Concatenate all dataframes into one
final_df = pd.concat(all_data, ignore_index=True)

# Optional: save to a new CSV
final_df.to_csv("combined_data.csv", index=False)

print("All CSVs have been combined successfully!")
