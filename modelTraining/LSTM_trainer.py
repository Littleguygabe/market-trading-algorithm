import pandas as pd
import os
import argparse
import numpy as np

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import joblib
from pathlib import Path


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

def trainModel(X,y,horizons):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    Xscaler = StandardScaler()
    X_train = Xscaler.fit_transform(X_train.reshape(-1,X_train.shape[-1])).reshape(X_train.shape)
    X_test = Xscaler.fit_transform(X_test.reshape(-1,X_test.shape[-1])).reshape(X_test.shape)

    yscaler = StandardScaler()
    y_train = yscaler.fit_transform(y_train)
    y_test_scaled = yscaler.transform(y_test)

        #build the model
    model = keras.models.Sequential([
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(128, return_sequences=False),
    ]) 
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    
    
    model.add(keras.layers.Dense(len(forecast_horizons)))

    initial_lr = 0.01
    optimiser = keras.optimizers.Adam(learning_rate = initial_lr)

    model.compile(optimizer = optimiser,
                loss='mae',
                metrics=['mse'],
                )

    callback = keras.callbacks.EarlyStopping(monitor='loss',patience=5)

    reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

    model.fit(X_train,
                        y_train,
                        epochs=50,
                        batch_size = 128,
                        validation_split = 0.1,
                        callbacks=[callback,reduce_lr_callback],
                        # verbose=1
                        )


    loss,mse = model.evaluate(X_test,y_test_scaled)

    return model,Xscaler,yscaler

def run(target_col,horizons):

    data = dataRead(target_col)
    X,y = panelData(data)
    model,X_scaler,Y_scaler = trainModel(X,y,horizons)


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
    run(target_column,forecast_horizons)

