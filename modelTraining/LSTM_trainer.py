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
    panel_df.dropna(inplace=True) # Add this line
    panel_vals = panel_df.values
    print(f"Shape of panelled data after cleaning: {panel_df.shape}")
    window_size = 75
    X,y = [],[]

    for i in range(window_size,len(panel_vals)):
        window = panel_vals[i-window_size:i] #format of (context window No rows) x ((No Features)*(No Tickers) columns) 
        target = panel_vals[i]

        X.append(window)
        y.append(target)


    X = np.array(X)
    y = np.array(y)

def dataRead(target_col):
    dataFolder = 'dataPipelineOutputData'

    df_list = []
    diagnostics = []

    for filename in os.listdir(dataFolder):
        try:

            filedf = pd.read_csv(os.path.join(dataFolder, filename), index_col='Date')
            
            # --- Start of new diagnostic code ---
            # Ensure index is datetime for min/max operations
            filedf.index = pd.to_datetime(filedf.index)
            
            diagnostics.append({
                'filename': filename,
                'row_count': len(filedf),
                'start_date': filedf.index.min(),
                'end_date': filedf.index.max()
            })
            # --- End of new diagnostic code ---

            filedf['Target'] = filedf[target_col].shift(-1) 
            filedf['Ticker'] = filename.split('.')[0]
            df_list.append(filedf)
            

        except Exception as e:
            print(f'ERROR > Error on {filename}: {e}')
            # print(f'Faulty Dataframe: {filedf}')


    # --- New: Print the diagnostic summary ---
    if diagnostics:
        summary_df = pd.DataFrame(diagnostics).sort_values(by='start_date', ascending=False)
        print("--- Data File Diagnostics (sorted by most recent start date) ---")
        with pd.option_context('display.max_rows', 200): # Ensure we see all files
             print(summary_df)
        print("----------------------------------------------------------------")


    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=['Target'])
    return df

def run(target_col):
    data = dataRead(target_col)
    # Temporarily disabling the paneling to focus on diagnostics
    # panelled_data = panelData(data)

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