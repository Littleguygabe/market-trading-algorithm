import os
import yfinance as yf
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing

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
        
    
    print('ERROR> Could not find the Ticker List file in Current, Child or Parent Directory')
    sys.exit(1)
    

def handleTxtTickerFile(ticker_file):
    file_path = getFileLocation(ticker_file)
    with open(file_path,'r') as f:
        contents = f.read()
        f.close()

    ticker_list = contents.split('\n')
    return ticker_list

def handleCsvTickerFile(ticker_file):
    file_path = getFileLocation(ticker_file)
    df = pd.read_csv(file_path)

    ticker_list = df['Ticker'].to_numpy()

    return ticker_list


def getTickerListFromFile(ticker_file):
    name,extension = os.path.splitext(ticker_file)
    if extension=='.txt':
        ticker_list = handleTxtTickerFile(ticker_file)

    elif extension=='.csv':
        ticker_list = handleCsvTickerFile(ticker_file)

    else:
        print(f'ERROR > File Extension must be `.txt` or `.csv` not: {extension}')
        sys.exit(1)

    return ticker_list

def flattenDfToArray(data,ticker_list,mdd):
    results = []
    witheld_ticker_arr = []

    for i in tqdm(range(len(ticker_list))):
        ticker = ticker_list[i]
        ticker_data = data.xs(ticker,level=1,axis=1).copy()
        ticker_data['Ticker'] = ticker
        ticker_data.dropna(inplace=True)
        if len(ticker_data)<mdd:
            witheld_ticker_arr.append({'Ticker':ticker,'N_Data_Points':len(ticker_data)})
            continue

        results.append(ticker_data)

    witheld_ticker_df = pd.DataFrame(witheld_ticker_arr)
    print('--- Missed Tickers due to Lack of Data Points --- ')
    with pd.option_context('display.max_rows',200):
        print(witheld_ticker_df)

    print(f'Number of Skipped Stocks is: {len(witheld_ticker_df)}')
    print(f'Final Number of Downloaded Stocks is: {len(results)}')

    print("----------------------------------------------------------------")

    return results

def getRawTickerData(ticker_list,mdd):
    df_arr = []
    ticker_list = ticker_list.tolist()
    tickers = yf.Tickers(ticker_list)
    data = tickers.download(period='1500d',interval='1d',progress=True)
    # need to re-format data into an array structure 
    
    df_arr = flattenDfToArray(data,ticker_list,mdd)
    return df_arr




def run(ticker_file,mdd):
    print('>--------------------')
    print('\033[1mRunning handleRawDataCollection.py\033[0m')
    print('Getting list of Tickers')
    ticker_list = getTickerListFromFile(ticker_file)
    print('Retrieved Ticker List')
    print('Getting raw data')
    raw_data_array = getRawTickerData(ticker_list,mdd)  
    print('\nAll raw data Retrieved')
    return raw_data_array
