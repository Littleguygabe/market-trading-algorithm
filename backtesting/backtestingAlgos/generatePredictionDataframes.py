import os
from pathlib import Path
import pandas as pd
import inquirer
from tqdm import tqdm
import multiprocessing
from functools import partial

# Absolute import from the project root to the sibling package
from modelPrediction import predictionPipeline as predPipeline
# Relative import to a module inside the current package

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

def init_worker(df):
    global worker_df
    worker_df = df

def workerPredictionFunction(end_idx,model_to_use,horizon):
    start_idx = 0

    working_df = worker_df.iloc[start_idx:end_idx]

    orderval,indicator = predPipeline.run(model_to_use,ticker=None,generate_order=True,horizon=horizon,verbose=-1,back_testing_data=working_df)

    # orderval,indicator = DOA()

    # print(orderval)
    # take the risk df and decide how much of the stock to buy
    # add the current Close price to the amount of the stock to buy and

    current_date = working_df.index[-1]
    current_close = working_df['Close'].iloc[-1]

    return_dict = {
        'idx':end_idx,
        'Date': current_date,
        'Close': current_close,
        'Indicator': indicator,
        'OrderVal': orderval,
        'Ticker': working_df['Ticker'].iloc[-1]
    }

    return return_dict

def getPredictionDataframe(df,model_to_use,horizon):
# parallelise the production of the prediction data frames -> use the same modules as the RFE parallelisation

    print('Initialising Parallelisation Algorithm')

    min_window_size = 75
    tasks = [i for i in range(min_window_size,len(df)+1)]
    num_cores = multiprocessing.cpu_count()
    #allows us to not have to pass in multiple arguments when 2/3 are constant
    partial_worker_func = partial(
        workerPredictionFunction,
        model_to_use=model_to_use,
        horizon=horizon
    ) 

    results = []
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(df,)) as pool:
        pbar = tqdm(pool.imap_unordered(partial_worker_func, tasks), total=len(tasks))
        for result in pbar:
            results.append(result)

    results.sort(key=lambda x: x['idx'])
    
    return pd.DataFrame(results)

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
        ticker = raw_data_df_arr[i]['Ticker'].iloc[1]
        print(f'Processing > {ticker}')
        df = raw_data_df_arr[i]
        pred_dataframe = getPredictionDataframe(df,model_to_use,horizon)
        predictions_df_arr.append(pred_dataframe)

    return predictions_df_arr 