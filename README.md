python -m backtesting.backtestingScript --data_folder backtestingdata -sc 100000 --horizon 5

first run the code to train the model unless xgboost model is trained, in which case put in the models folder along with params

modelTraining/datapipeline
args: 
    --ticker_file -> file containing the tickers to be trained on
    --target -> the target column that will be used for the model

model training/xgboost trainer run 1st
args:
    horizons -> list of hoirzons
