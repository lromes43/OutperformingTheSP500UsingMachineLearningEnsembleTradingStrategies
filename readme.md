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
binaryone = joblib.load("/Users/lukeromes/Desktop/Personal/Sp500Project/Models/FinalBoostedOneDayClassifier.joblib")
continuousone = joblib.load("/Users/lukeromes/Desktop/Personal/Sp500Project/Models/ContinuousOneDayFinal.job.lib")
```

Preparing the Data for the One-Day Binary Classifier and running the
model

``` python
import pandas as pd
import xgboost as xgb

drop_cols = ['Date', 'next_day_pct_change','Daily_Return',
 'next_5_day_pct_change',
 'Movement_5_day',
 'next_30_day_pct_change',
 'Movement_30_day',
 'Movement']

X = test.drop(drop_cols, axis = 1)
X_final = pd.get_dummies(X, drop_first=True)
actual = test['Movement'].astype(int)

dtest = xgb.DMatrix(X_final, label = actual)

pred = binaryone.predict(dtest)
pred_final = (pred >=.5).astype(int)

cm = confusion_matrix(actual, pred_final)
finalboostedcm = ConfusionMatrixDisplay(confusion_matrix=cm)
finalboostedcm.plot(cmap = "Blues")
plt.title("Final One Day Binary Confusion Matrix")
final_boosted_acc = accuracy_score(actual, pred_final)
print(final_boosted_acc)





pred_final_df = pd.DataFrame(pred_final)
pred_final_df = pred_final_df.reset_index().rename(columns={'index': 'iteration', 0: 'actual up/down'})
actual = actual.reset_index().rename(columns={'index': 'iteration'})
merged_binary = pd.merge(pred_final_df, actual, how='inner', on='iteration')
sorted_ticker_series = transformed_data['Ticker'].sort_values(ascending=True)
merged_binary['ticker'] = sorted_ticker_series.reset_index(drop=True)
merged_binary_one = merged_binary
plt.show()
```

    0.7325354969574036

![](readme_files/figure-commonmark/cell-3-output-2.png)

    hi

``` python
import pandas as pd
```
