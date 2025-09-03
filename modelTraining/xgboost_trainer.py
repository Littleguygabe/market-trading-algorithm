import xgboost as xgb
import pandas as pd
import os
import argparse
from pathlib import Path
import joblib
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split


def dataRead(target_col):
    dataFolder = 'dataPipelineOutputData'

    df_list = []
    df = pd.DataFrame()

    for filename in os.listdir(dataFolder):
        try:

            filedf = pd.read_csv(os.path.join(dataFolder, filename), index_col=0)
            filedf['Target'] = filedf[target_col].shift(-1) 
            

            df_list.append(filedf)

        except Exception as e:
            print(f'ERROR > Error on {filename}: {e}')
            print(f'Faulty Dataframe: {filedf}')


    df = pd.concat(df_list,join='inner', ignore_index=True)
    df = df.dropna()
    print(df)
    return df

def run(target_Col):
    
    df = dataRead(target_Col)

    X = df.drop(columns=['Target','Date'])
    y = df['Target']

    print(f'Training dataset: {X}')

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=3,
        learning_rate=0.1,
        eval_metric='rmse',
        n_estimators=1000
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False 
    )
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print('>--------------------')
    print('XGboost Algorithm Finished Training')
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared: {r2}")

    return model,X.columns,r2

def saveModel(model,forecast_horizons,input_features,target_Col,r2):
    print('>--------------------')
    modelSavePrefix=''
    for horizon in forecast_horizons:
        modelSavePrefix+=str(horizon)

    save_folder = modelSavePrefix+'_xgBoost_'+target_Col+'-r2_'+str(round(r2,3))
    script_path = Path(__file__)
    root_path = script_path.parent
    root_parent_dir = root_path.parent
    parent_dir = root_parent_dir/'models'
    save_dir = parent_dir/save_folder
    count = 0
    while save_dir.is_dir():
        count+=1
        save_dir = parent_dir/f'{save_folder}_{count}'

    print(f'Saving model to > {save_dir}')

    save_dir.mkdir(parents=True,exist_ok=True)
    model.save_model(save_dir/'model.json')
    with open(f'{save_dir}/finalRFEresults.txt','w') as f:
        for feature in input_features:
           f.write(f'{feature}\n')

        f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an LSTM model to forecast stock prices.')
    parser.add_argument('--horizons',nargs='+', type=int, default=1, help='The forecast horizon in days.')
    parser.add_argument('--save_model', action='store_true', help='Flag to save the trained model.')
    parser.add_argument('--target',required=True,help='The target column to predict')
    args = parser.parse_args()

    forecast_horizons = args.horizons
    should_save_model = args.save_model
    target_column = args.target
    print('>--------------------')
    print('Running XGboost Algorithm')
    model,input_features,r2 = run(target_column)

    if should_save_model:
        saveModel(model,forecast_horizons,input_features,target_column,r2)
    print('>--------------------')
