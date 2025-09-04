import pandas as pd
import os
import argparse
import numpy as np


def panelData(df):
    columns = df.columns.values

    #remove columns being used for pivot and index from features 
    isin_mask = ~np.isin(columns,['Date','Ticker'])
    columns = columns[isin_mask]

    panel_df = df.pivot(index='Date',columns='Ticker',values=columns)
    panel_df.ffill(inplace=True)
    panel_df.dropna(inplace=True)
    panel_vals = panel_df.values
    window_size = 75
    X,y = [],[]

    for i in range(window_size,len(panel_vals)):
        window = panel_vals[i-window_size:i] #format of (context window No rows) x ((No Features)*(No Tickers) columns) 
        target = panel_vals[i]

        X.append(window)
        y.append(target)


    X = np.array(X)
    y = np.array(y)

    # shape of (N samples, Context size, N_features*N_tickers) -> this allows us to take the whole market as context rather than just a single ticker's current environment
    #   can think of as adding horizontal aswell as vertical context
    # print(X.shape)

    return X,y

def dataRead(target_col):
    dataFolder = 'dataPipelineOutputData'

    df_list = []
    df = pd.DataFrame()

    for filename in os.listdir(dataFolder):
        try:

            filedf = pd.read_csv(os.path.join(dataFolder, filename), index_col=0)
            filedf['Target'] = filedf[target_col].shift(-1) 
            filedf['Ticker'] = filename.split('.')[0]
            df_list.append(filedf)
            

        except Exception as e:
            print(f'ERROR > Error on {filename}: {e}')
            print(f'Faulty Dataframe: {filedf}')


    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=['Target'])
    return df

def run(target_col):
    data = dataRead(target_col)
    X,y = panelData(data)

    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train an LSTM model to forecast stock prices.')
    parser.add_argument('--horizons',nargs='+', type=int, default=1, help='The forecast horizon in days.')
    parser.add_argument('--save_model', action='store_true', help='Flag to save the trained model.')
    parser.add_argument('--target',required=True,help='The target column to predict')
    args = parser.parse_args()

    forecast_horizons = args.horizons
    should_save_model = args.save_model
    target_column = args.target
    print('>--------------------')
    print('Running LSTM Algorithm')
    run(target_column)

