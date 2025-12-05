# Optimizing Stock Market Performance Using Ensemble Approaches of
Machine Learning


## Introduction

The outcome of this project is multi-fold, the goal is to predict if a
stock will go up or down in the future across different time horizons
and by what percent will the future stock price increase or decrease.
The success of the model will be determined by not only its ability to
correctly predict price movements but also its ability to act on them.
This will be done by the model acting as a hedge fund manager of a
financial firm trying to outperform the SP500 baseline.

## Code

``` python
import pandas as pd
import joblib
import pandas as pd 
import numpy as np   
from plotnine import *  
import xgboost as xgb 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
)
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm 
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
test = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/TestData.feather")
transformed_data = test
```

``` python
print('hello_world')
```

    hello_world

    hi

``` python
import pandas as pd
```
