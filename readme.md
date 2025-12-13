# Optimizing Stock Market Performance Using ML Ensemble Models


# Please reach out to me at romesluke@gmail.com for model access.

# Introduction

The outcome of this project is multi-fold, the goal is to predict if a
stock will go up or down in the future across different time horizons
(1, 5 and 30 Days) and by what percent will the future stock price
increase or decrease. The success of the model will be determined by not
only its ability to correctly predict price movements but also its
ability to act on them. This will be done by the model acting as a hedge
fund manager of a financial firm trying to outperform the SP500
baseline.

# Outcome

This model consistently outpreforms the SP500 on a multitude of
different range of dates. From September 2nd to November 11th, this
model achieved a 35% return compared to the SP500 return of 3%. To test
its effectiveness on future dates not included in the model, this model
achieved a 14% return from November 12 through November 28th compared to
the SP500 return of -0.29%. See graphs at the end. of this document as
well as the Trading Framework Architecture included in the SP500
Comparsion subfolder.

## Data

Data used was pulled directly from yahoo finance API (yfinance). Pulled
data included every SP500 stock from November 10th, 2024 through
November 11th, 2025. Initial variables pulled included:(Date, Price,
Close, High, Low, Open, Volume, Ticker). These first initial features
were included so the model would be able to uniquely identify each stock
and its corresponding price via the composite key of Ticker and Date.
The initial eight variables were expanded upon resulting in a total of
46 variables (6 output, 40 input). This data was then exported to a
feather file to improve performance due to its binary, columnar format
allowing for faster processing speeds and reduced storage needs. You can
find the final data to run this in the data subfolder entitled
“CORRECTEDFINALDATA.feather”. Additionally you can use the scripts
within the DataLoadingScripts folder (work in progress in turning into a
full pipeline) which calls the yfinance API to retrieve the data,
transforms it by creating additional derived variables, and implements a
train test split.

## Machine Learning Models

Machine learning models are pretrained and saved in the Models subfolder
as .joblib files. To utilize this you must do below, and replace the
file path with your desired model. Additionally you may recreate or
retrain the models that I have created which can be found in the ML
subfolder within the Scripts subfolder. These scripts are saved as qmd,
allowing for them to be run cell by cell.

``` python
import joblib
binaryone = joblib.load("/Users/lukeromes/Desktop/Personal/Sp500Project/Models/FinalBoostedOneDayClassifier.joblib")
```

## Running the Machine Learning Models

In order to run the models you have two options: utilize the predefined
functions found in the functions subfolder entitled
“binary_preprocessing_func” and “cont_preprocessing_func” which
transform the derived variable data into the final input form needed to
be understood by the models and executes and saves the results. Both of
these models require three arguments: “date”: the date which you want to
predict, “file_path” where is the derived variables data being pulled
from, and “Predictor” is trying to be predicted.

The second option involves going to the “Model Comparions” subfolder
with the “Scripts” subfolder and running the PerformanceOnTest( 1 OR 5
OR 30)Day scripts. These were the original scipts that were created
before functions were created. A benefit to this method is that these
scripts output the results and model performance.

## Model Simulation vs Real Data

The final objective that this model set out to accomplish was how well
it performs in a trading environment. In this study a simulated trading
environment was used with two different time periods, the testing period
as well as a range of future dates which the model had not been trained
nor previously been tested on. Testing dates for the simulation include
September 3rd, 2025 through November 7th 2025 while the new data
includes November 11th, 2025 through November 26th, 2025.

The architecture powering this simulation is quite simple as each model,
the binary and continuous run simulateneously and output and store their
results. A temporary table for each model is created to hold the results
along with a dummy variable called “Buy” which is 1 if the model
predicts future positive movement and 0 if the model does not. These
temporary tables are then subsetted to only show instances where the buy
column is equal to one and the two tables, temporary binary and
temporary continuous are merged on date and ticker ensuring the final
results are only stocks of which were predicted to go up by both models.
Next, the final merged table is sorted based on predicted percent change
and the top 10 stocks are extracted and evenly bought from the initial
\$100,000 starting capital. This process then repeats for the remainder
of the dates, but checks if each day’s predictions align with current
holdings. If the predictions align with the current holdings, the model
continues, if the current holdings do not match the model predictions,
the current holdings are sold. The shares and selling price is extracted
and added to a variable called cash which is then used to invest in new
securities. (Detailed Diagram can be found in SP500 Comparison
Subsection)

To accomplish this proceed to the SP500 Comparison subfolder.First open
the “SP500 Comparsion subfolder” and open
ModelPerformanceComparedToSp500Test, this scrript will run a
“simulation” as described above and buy and sell stocks using pretend
capital. Once this is finished running open the
CreatingDataforModelResultsFuture (again working on making more
streamlined), this will transform the results of the model to a
plottable format as well as pull the SP500 prices from the same days
allowing for the model performance to truly be evaluated.

### Trading Simulation Results vs SP500

Here I am showing the model results on future data Nov 11th - Nov 26th
2025.

``` python
model_perf = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/Result Data/modelovertime_future.csv")
removed_holdings = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/Result Data/removed_holdings_future.csv")
trade_log = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/Result Data/trade_log_df_future.csv")
sp_final = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/Result Data/sp500_final.csv")
sp_prices = pd.read_csv("/Users/lukeromes/Desktop/Personal/Sp500Project/SP500Comparison/Result Data/sp500closing.csv")
trade_log.loc[0, 'Cash_before'] = 100000
import numpy as np
future_merged = pd.merge(removed_holdings, trade_log, how= 'inner', left_on=['Stock', 'Shares_Owned'], right_on=['Ticker', 'Shares'])
future_merged = future_merged.drop('Unnamed: 0', axis = 1)
sp_prices['shares'] = 100000 / sp_prices['Close'].iloc[0]
sp_prices['value'] = np.nan
for i in range(0, len(sp_prices)):
    sp_prices['value'].iloc[i] = sp_prices['shares'].iloc[0] * sp_prices['Close'].iloc[i]


final_sp_future = sp_prices
final_sp_future['Date'] = pd.to_datetime(final_sp_future['Date'])
final_sp_future = final_sp_future[final_sp_future['Date'] >= '2025-11-12']
final_sp_future = final_sp_future[final_sp_future['Date'] <= '2025-11-26']
trade_log['Date'] = pd.to_datetime(trade_log['Date'])
from plotnine import *
future_price_comparisons_plot = (ggplot(trade_log, aes(x = 'Date', 
                                                        y = 'Cash_before'))+ geom_line()+ geom_line(final_sp_future, aes(x ='Date',y = 'value' ), color = 'red'))
trade_log['label'] = 'Model Trading'
final_sp_future['label'] = 'S&P 500 Baseline'

test_price_comparisons_plot = (
    ggplot() +
    geom_line(
        trade_log,
        aes(x='Date', y='Cash_before', color='label')
    ) +
    geom_line(
        final_sp_future,
        aes(x='Date', y='value', color='label')
    ) +
    scale_color_manual(values=['navy', 'dodgerblue']) +
    scale_x_datetime(date_breaks="3 days", date_labels="%Y-%m-%d") +
    labs(
        color='Legend',
        y='Value',
        title='Model Trading vs SP500 During Future Period'
    ) +
    theme(
        axis_text_x=element_text(rotation=45, hjust=1),
        figure_size=(12, 6)
    )
)

test_price_comparisons_plot
```

<img src="readme_files/figure-commonmark/cell-8-output-1.png"
width="576" height="288" />

``` python
print(future_merged)
```

      Stock   Sell_Date  Sell_Price  Shares_Owned       Proceeds     sell_value  \
    0  GEHC  2025-11-13   73.699997   1085.481727   80000.000000   80000.000000   
    1  SOLV  2025-11-14   73.510002   1362.052423  100124.476529  100124.476529   
    2  AMAT  2025-11-18  223.761387    492.371724  110173.779582  110173.779582   
    3  AMCR  2025-11-19    8.440000  13053.766002  110173.779582  110173.779582   
    4  AMTM  2025-11-20   21.750000   5065.461130  110173.779582  110173.779582   
    5  GEHC  2025-11-21   73.129997   1506.547022  110173.779582  110173.779582   
    6    MU  2025-11-25  213.410004    538.668069  114957.154669  114957.154669   
    7  AMTM  2025-11-26   28.350000   4054.926036  114957.154669  114957.154669   

      Action_x        Date Ticker Action_y       Price        Shares  \
    0     Sell  2025-11-12   GEHC      BUY   73.699997   1085.481727   
    1     Sell  2025-11-13   SOLV      BUY   73.510002   1362.052423   
    2     Sell  2025-11-14   AMAT      BUY  203.351394    492.371724   
    3     Sell  2025-11-18   AMCR      BUY    8.440000  13053.766002   
    4     Sell  2025-11-19   AMTM      BUY   21.750000   5065.461130   
    5     Sell  2025-11-20   GEHC      BUY   73.129997   1506.547022   
    6     Sell  2025-11-21     MU      BUY  204.529999    538.668069   
    7     Sell  2025-11-25   AMTM      BUY   28.350000   4054.926036   

         Cash_before  
    0  100000.000000  
    1  100124.476529  
    2  100124.476529  
    3  110173.779582  
    4  110173.779582  
    5  110173.779582  
    6  110173.779582  
    7  114957.154669  

Simulation two consisting of unseen future dates outpreformed the SP500.
The model resulted in a 14.958% return compared to -0.29% return that
the SP500 returned during the same time period. During this simulation,
the model purchases 18 securities and sells 9.

## Next Steps

Retrain this model and prepare for it to go live.

In the functions folder save each function that has been created and
implement the creating derived vars func, initial data loading func, and
train test split func into an ETL pipeline. This will eliminate the need
for the data subfolder. The results will still be outputted as a pickle
file to improve preformance, but will not need to physically live on the
device.

Save the remaining two functions, binary preprocessing func, and cont
preprocessing func and turn those into one function which automatically
transforms the data into its final input form for the model, runs the
models, and saves the results. I hope to have these tasks done by Dec
20th, 2025.
