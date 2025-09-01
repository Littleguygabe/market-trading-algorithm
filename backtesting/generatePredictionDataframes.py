import os
from pathlib import Path
import pandas as pd
import inquirer
from tqdm import tqdm

# Absolute import from the project root to the sibling package
from modelPrediction import predictionPipeline as predPipeline
# Relative import to a module inside the current package
from .backtestingAlgos.loading_bar import loading_bar

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

    dirs_to_check.append(os.path.join(project_dir,'backTestingData')) #check if its in the shared data folder

    for directory in dirs_to_check:
        current_path = os.path.join(directory,file_name)
        print(f'checking > {current_path}')
        if os.path.exists(current_path):
            return current_path

def getModelList():
    models = os.listdir(getFileLocation('models'))
    return models


def readDirectory(data_path):
    raw_df_arr = []
    for file in os.listdir(data_path):
        data = pd.read_csv(os.path.join(data_path,file))
        ticker,extension = os.path.splitext(file)        
        data['Ticker'] = ticker
        raw_df_arr.append(data)

    return raw_df_arr

def getPredictionDataframe(df,model_to_use,horizon):
    output_arr = []
    for i in tqdm(range(201,len(df))):
        working_df = df[:i]

        #pass the working df into the prediction pipeline to get a risk data frame
        orderval,indicator = predPipeline.run(model_to_use,ticker=None,generate_order=True,horizon=horizon,verbose=-1,back_testing_data=working_df)
        # print(orderval)
        # take the risk df and decide how much of the stock to buy
        # add the current Close price to the amount of the stock to buy and

        current_date = df.index[i]
        current_close = df['Close'].iloc[i]

        output_arr.append({
            'Date': current_date,
            'Close': current_close,
            'Indicator': indicator,
            'OrderVal ($)': orderval,
            'Ticker': df['Ticker'].iloc[i]
        })


    # output df in format:
        # Date Close Indicator -1/1 (sell/buy) OrderVal ($) Ticker

    return pd.DataFrame(output_arr)

def run(data_folder,starting_cap,horizon):
    data_path = getFileLocation(data_folder)
    raw_data_df_arr = readDirectory(data_path)
    cap_per_stock = starting_cap/len(raw_data_df_arr)

    #get the model to use for the predictions
    model_choice = [inquirer.List('selection',
                                    message='Choose an LSTM Model to use:',
                                    choices=getModelList())]
    model_to_use = inquirer.prompt(model_choice)['selection']

    # get a prediction for every day in the dataframe
    predictions_df_arr = []
    print('Generating Predictions on the Provided Historical data')
    for i in range(len(raw_data_df_arr)):
        print(raw_data_df_arr[i]['Ticker'].iloc[1])
        df = raw_data_df_arr[i]
        pred_dataframe = getPredictionDataframe(df,model_to_use,horizon)
        predictions_df_arr.append(pred_dataframe)

    print(predictions_df_arr)
    quit()