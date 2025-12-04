---
title: "Untitled"
format: html
---


```{python}
import joblib
import pandas as pd
from plotnine import *
```


```{python}
train = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/TrainData.feather")
test = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/TestData.feather")

X_train_raw = train.drop([
    'Daily_Return','next_day_pct_change','next_5_day_pct_change',
    'Movement_5_day','next_30_day_pct_change','Movement_30_day','Movement'
], axis=1)

X_train_raw = pd.get_dummies(X_train_raw, drop_first=True)

X_train_final = X_train_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

y_train_final = train['next_day_pct_change'].astype(float)

dtrain1 = xgb.DMatrix(X_train_final, label=y_train_final)




test_clean = test.dropna(subset=['next_day_pct_change']).copy()

X_test_raw = test_clean.drop([
    'Daily_Return','next_day_pct_change','next_5_day_pct_change',
    'Movement_5_day','next_30_day_pct_change','Movement_30_day','Movement'
], axis=1)

X_test_raw = pd.get_dummies(X_test_raw, drop_first=True)
X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)

X_test_final = X_test_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

y_test_final1 = test_clean['next_day_pct_change'].astype(float)

dtest1 = xgb.DMatrix(X_test_final, label=y_test_final1)
```

```{python}
train = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/TrainData.feather")
test = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/TestData.feather")

X_train_raw = train.drop([
    'Daily_Return','next_day_pct_change','next_5_day_pct_change',
    'Movement_5_day','next_30_day_pct_change','Movement_30_day','Movement'
], axis=1)

X_train_raw = pd.get_dummies(X_train_raw, drop_first=True)

X_train_final = X_train_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

y_train_final = train['next_5_day_pct_change'].astype(float)

dtrain2 = xgb.DMatrix(X_train_final, label=y_train_final)




test_clean = test.dropna(subset=['next_5_day_pct_change']).copy()

X_test_raw = test_clean.drop([
    'Daily_Return','next_day_pct_change','next_5_day_pct_change',
    'Movement_5_day','next_30_day_pct_change','Movement_30_day','Movement'
], axis=1)

X_test_raw = pd.get_dummies(X_test_raw, drop_first=True)
X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)

X_test_final = X_test_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

y_test_final5 = test_clean['next_5_day_pct_change'].astype(float)

dtest2 = xgb.DMatrix(X_test_final, label=y_test_final5)
```


```{python}

train = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/TrainData.feather")
test = pd.read_feather("/Users/lukeromes/Desktop/Personal/Sp500Project/Data/TestData.feather")

X_train_raw = train.drop([
    'Daily_Return','next_day_pct_change','next_5_day_pct_change',
    'Movement_5_day','next_30_day_pct_change','Movement_30_day','Movement'
], axis=1)

X_train_raw = pd.get_dummies(X_train_raw, drop_first=True)

X_train_final = X_train_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

y_train_final = train['next_30_day_pct_change'].astype(float)

dtrain3 = xgb.DMatrix(X_train_final, label=y_train_final)




test_clean = test.dropna(subset=['next_30_day_pct_change']).copy()

X_test_raw = test_clean.drop([
    'Daily_Return','next_day_pct_change','next_5_day_pct_change',
    'Movement_5_day','next_30_day_pct_change','Movement_30_day','Movement'
], axis=1)

X_test_raw = pd.get_dummies(X_test_raw, drop_first=True)
X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)

X_test_final = X_test_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

y_test_final30 = test_clean['next_30_day_pct_change'].astype(float)

dtest3 = xgb.DMatrix(X_test_final, label=y_test_final30)

```



```{python}
one_day_model = joblib.load('/Users/lukeromes/Desktop/Personal/Sp500Project/Models/ContinuousOneDayFinal.joblib')
five_day_model = joblib.load('/Users/lukeromes/Desktop/Personal/Sp500Project/Models/ContinuousFiveDayFinal.joblib')
thirty_day_model = joblib.load('/Users/lukeromes/Desktop/Personal/Sp500Project/Models/ContinuousThirtyDayFinal.joblib')
```


one day predictions

```{python}
one_pred = one_day_model.predict(dtest1)
```

```{python}
y_test_final1 = (
    y_test_final1
        .reset_index()                       
        .rename(columns={'next_day_pct_change':'actual_one_day'})
)

y_test_final1['one_day_prediction'] = one_pred
y_test_final1 = y_test_final1.drop('index', axis=1)

```

one day MSE

```{python}
error = []
for i in range(0, len(y_test_final1+1)):
    diff = float(y_test_final1['actual_one_day'].iloc[i] - y_test_final1['one_day_prediction'].iloc[i]) **2
    error.append(diff)
    Final_MSE_1= (1/len(merged)) * sum(error)
print(Final_MSE_1)
```

One Day MSE: 0.00045527

one day RMSE

```{python}
one_day_rmse= Final_MSE_1**.5
print(one_day_rmse)
```

One Day RMSE: 0.021337086


five day prediction

```{python}
five_pred = five_day_model.predict(dtest2)
```

```{python}
y_test_final5 = (
    y_test_final5
        .reset_index()                       
        .rename(columns={'next_5_day_pct_change':'actual_five_day'})
)

y_test_final5['five_day_prediction'] = five_pred
y_test_final5 = y_test_final5.drop('index', axis=1)

```


five day MSE

```{python}
error = []
for i in range(0, len(y_test_final5+1)):
    diff = float(y_test_final5['actual_five_day'].iloc[i] - y_test_final5['five_day_prediction'].iloc[i]) **2
    error.append(diff)
    Final_MSE_5= (1/len(merged)) * sum(error)
print(Final_MSE_5)
```

Five Day MSE: 0.00204304

five day RMSE

```{python}
five_day_rmse= Final_MSE_5**.5
print(five_day_rmse)
```

Five DAY RMSE: 0.0452000



thirty day prediction

```{python}
thirty_pred = thirty_day_model.predict(dtest3)
```

```{python}
y_test_final30 = (
    y_test_final30
        .reset_index()                       
        .rename(columns={'next_30_day_pct_change':'actual_thirty_day'})
)

y_test_final30['thirty_day_prediction'] = thirty_pred
y_test_final30 = y_test_final30.drop('index', axis=1)

```



thirty day MSE

```{python}
error = []
for i in range(0, len(y_test_final30+1)):
    diff = float(y_test_final30['actual_thirty_day'].iloc[i] - y_test_final30['thirty_day_prediction'].iloc[i]) **2
    error.append(diff)
    Final_MSE_30= (1/len(merged)) * sum(error)
print(Final_MSE_30)
```

Thirty Day MSE: 0.004933448


thirty day RMSE


```{python}
thirty_day_rmse= Final_MSE_30**.5
print(thirty_day_rmse)
```

thirty day RMSE: 0.0703095


Creating a Dataframe so can compare RMSE and MSE and across time periods

```{python}
comparison_df_MSE = pd.DataFrame({
    'one_day_mse': Final_MSE_1,
    'five_day_mse': Final_MSE_5,
    'thirty_day_mse':Final_MSE_30,
},index=['Model Performance'])
```

```{python}
comparison_df_RMSE = pd.DataFrame({
    'one_day_RMSE': one_day_rmse,
    'five_day_RMSE':five_day_rmse,
    'thirty_day_RMSE':thirty_day_rmse
},index=['Model Performance'])
```

```{python}
melted_comparison_df_MSE =pd.melt(comparison_df_MSE )
melted_comparison_df_RMSE = pd.melt(comparison_df_RMSE)
```

Comparing MSE across time periods

```{python}
MSE_plot = (ggplot(melted_comparison_df_MSE, aes(x = 'value'))+
            geom_bar())
```


```{python}

MSE_plot = (
    ggplot(melted_comparison_df_MSE, aes(x='variable', y='value'))
    + geom_bar(stat='identity') 
    + labs(
        x='Forecast Horizon',
        y='Mean Squared Error (MSE)',
        title='Comparison of MSE across Forecast Horizons'
    )
    + aes(fill='variable') 
)


```

It makes sense get the lowest MSE with the one day prediction and highest with one month

```{python}
RMSE_plot = (
    ggplot(melted_comparison_df_RMSE, aes(x='variable', y='value'))
    + geom_bar(stat='identity') 
    + labs(
        x='Forecast Horizon',
        y='Root Mean Squared Error (RMSE)',
        title='Comparison of RMSE across Forecast Horizons'
    )
    + aes(fill='variable')  + 
    theme_minimal()
)
```

It makes sense get the lowest RMSE with the one day prediction and highest with one month
