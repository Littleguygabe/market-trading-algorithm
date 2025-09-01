import argparse
import inquirer
import os
from pathlib import Path
import sys
import yfinance as yf
from .predictionAlgos import featureEngineering
from .predictionAlgos import GARCHpipeline
from .predictionAlgos import xgboostPrediction as xgbp
from .predictionAlgos import monteCarloSimulation as MCS
from .orderSizingAlgos import riskRatio as orderFunction


def getFileLocation(file_name):
    if os.path.exists(file_name):
        return file_name
    
    dirs_to_check = []
    script_dir = Path(__file__) #already checked
    root_dir = script_dir.parent #same level as datapipelinefolder

    dirs_to_check.append(root_dir)

    root_parent_dir = root_dir.parent
    dirs_to_check.append(root_parent_dir) #check if its main project dir

    project_dir = root_parent_dir.parent
    dirs_to_check.append(project_dir)

    dirs_to_check.append(os.path.join(project_dir,'shareddata')) #check if its in the shared data folder

    for directory in dirs_to_check:
        current_path = os.path.join(directory,file_name)
        print(f'checking > {current_path}')
        if os.path.exists(current_path):
            return current_path
        
    
    print('ERROR > Could not find the Ticker List file in Current, Child or Parent Directory')
    sys.exit(1)

def getModelList():
    models = os.listdir(getFileLocation('models'))
    return models

def getRawData(ticker):
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period='200d',interval='1d')
    return data

def getInputFeatures(model_to_use_path):
    with open(os.path.join(model_to_use_path,'finalRFEresults.txt'),'r') as f:
        contents = f.read()
        f.close()

    features = contents.split('\n')[:-1]
    return features

def run(model_to_use,ticker,generate_order,horizon=30,num_sims=5000,verbose=1,back_testing_data = None,capital_per_trade=None):
    BOLD = '\033[1m'
    END = '\033[0m'

    if verbose>-1:
        print('>--------------------')
        print('Generating Return Prediction')

    model_to_use_path = os.path.join(getFileLocation('models'),model_to_use)

    # [x] download data for the specified ticker - 200 days for the garch pipeline
    # [x] feature engineer the data 
    # [x] add the GARCH volatility prediction (same workflow as in the data pipeline)
    # [x] read the features to be used from the specified model directory
    # [x] extract the specified features from the features dataframe
    # [x] get a single row of the produced data features
    # [x] pass that into the model to produce a return prediction
    # [x] take return prediction from xgxboost model & the volatility prediction from the GARCH pipeline then pass into mcs pipeline for data output
    #   ^ volatility prediction needs to be the prediction for the next day

    if back_testing_data is not None:
        raw_stock_df = back_testing_data

    else:
        raw_stock_df = getRawData(ticker)

    non_vol_feature_df = featureEngineering.run(raw_stock_df,verbose)
    vol_feature_df,vol_forecast_val = GARCHpipeline.run(non_vol_feature_df,verbose)
    return_pred_input_features = getInputFeatures(model_to_use_path)
    return_pred = xgbp.run(model_to_use_path,vol_feature_df[return_pred_input_features],verbose)

    last_close_price = raw_stock_df.tail(1)[['Close']].values

    price_paths = MCS.run(return_pred,vol_forecast_val,int(horizon),int(num_sims),last_close_price,verbose)
    risk_df = MCS.displayPriceAnalytics(price_paths,last_close_price,verbose)

    if generate_order:
        order_val,indicator = orderFunction.run(risk_df,capital_per_trade,last_close_price)
        return order_val,indicator
    else:
        return risk_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prediction Pipeline')
    parser.add_argument('--model',help='Pre-Specify a Model to use if you Already know the folder name')
    parser.add_argument('--horizon',help='Number of days to Simulate',default=30)
    parser.add_argument('--num_sims',help='Number of Simulations to run',default=5000)
    parser.add_argument('--verbose',
                        default='0',
                        const='0',
                        nargs = '?',
                        help = '0 - nothing printed, 1 - print the risk dataframe, 2 - print risk and draw graphs',
                        type=int) 

    parser.add_argument('--ticker',help='Ticker for the Analysis to be ran on')
    parser.add_argument('--generate_order',action='store_true',help='Use the Prediction to Generate a buy or sell Order')
    parser.add_argument('--max_loss_per_trade','-mlpt','-cpt',help='The Amount of Capital you are Willing to risk per Trade',type=float)

    args = parser.parse_args()

    if args.generate_order and not args.max_loss_per_trade:
        parser.error('the --max_loss_per_trade (-mlpt) flag is requied when --generate_order is used')

    if args.model:
        model_to_use = args.model

    else:
        model_choice = [inquirer.List('selection',
                                     message='Choose an LSTM Model to use:',
                                     choices=getModelList())]
        print('>--------------------')
        model_to_use = inquirer.prompt(model_choice)['selection']
    
    run(model_to_use,args.ticker,args.generate_order,args.horizon,args.num_sims,args.verbose,None,args.max_loss_per_trade)