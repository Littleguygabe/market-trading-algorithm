import pandas as pd
import os
import argparse


def dataRead(target_col):
    dataFolder = 'dataPipelineOutputData'

    df_list = []
    df = pd.DataFrame()

    for filename in os.listdir(dataFolder):
        try:

            filedf = pd.read_csv(os.path.join(dataFolder, filename), index_col=0)
            filedf['Target'] = filedf[target_col].shift(-1) 
            # filedf = filedf.dropna()
            # print(filedf)
            df_list.append(filedf)
            

        except Exception as e:
            print(f'ERROR > Error on {filename}: {e}')
            print(f'Faulty Dataframe: {filedf}')


    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=['Target'])
    return df

def run(target_col):
    data = dataRead(target_col)
    print(data)

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

