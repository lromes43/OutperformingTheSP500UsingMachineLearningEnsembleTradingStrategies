
import pytest
import pandas as pd
import os
# Import your functions
from Functions.sp500_pipeline import run_sp500_pipeline
from Functions.split import train_test_split_by_date_function

def test_sp500_pipeline_full_execution():
    """Step 1: Run the actual pipeline logic to generate the feather file."""
    start_date = "2025-11-01"
    end_date = "2025-12-15"
    
    run_sp500_pipeline(start_date, end_date)
    
    output_path = "Data/FINALSP500Data.feather"
    assert os.path.exists(output_path), "Pipeline failed to create the feather file"
    
    df = pd.read_feather(output_path)
    assert len(df) > 0

def test_pipeline_data_quality():
    """Step 2: Audit the data for 'Accuracy' in variable creation."""
    df = pd.read_feather("Data/FINALSP500Data.feather")
    

    completeness = (1 - df.isnull().sum().sum() / df.size) * 100
    variable_count = len(df.columns)
    
    print(f"\n--- Pipeline Quality Audit ---")
    print(f"Variable Creation Accuracy: {completeness:.2f}% Complete")
    print(f"Total Factors Engineered: {variable_count}")
    
    assert completeness > 70, "Data quality too low (too many NaNs)"
    assert variable_count >= 30, "Missing technical factors"

def test_split_logic():
    """Step 3: Testing Train/split."""
    data_path = "Data/FINALSP500Data.feather"
    train_start_date = "2024-11-01"
    train_end_date = "2025-10-08"
    test_start_date = "2025-10-09"
    
    train, test = train_test_split_by_date_function(data_path, train_start_date, train_end_date, test_start_date)
    
    assert not train.empty, "Training set is empty"
    assert not test.empty, "Testing set is empty"
    assert train['Date'].max() <= pd.to_datetime(train_end_date), "Leakage: Train date > train_end"
