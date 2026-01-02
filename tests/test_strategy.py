import pytest
import pandas as pd
import os
from Functions.sp500_pipeline import run_sp500_pipeline

def test_sp500_pipeline_full_execution():
    """
    runs the code in sp500_pipeline.py.
    """
    start_date = "2024-01-01"
    end_date = "2024-01-05"
    
 
    run_sp500_pipeline(start_date, end_date)
    

    output_path = "Data/FINALSP500Data.feather"
    assert os.path.exists(output_path), "Pipeline failed to create the feather file"
    

    df = pd.read_feather(output_path)
    assert len(df) > 0