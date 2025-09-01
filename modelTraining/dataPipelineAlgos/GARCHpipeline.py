from arch import arch_model
import os
from dataPipelineAlgos.loading_bar import loading_bar
import sys
import pandas as pd

def generateVolatilityPrediction(df):
    output_df = df.copy()
    y_scaler = 100
    try:
        output_df['GARCH_scaled_return'] = output_df['Return']*y_scaler
        model = arch_model(output_df['GARCH_scaled_return'].dropna(),vol='GARCH',q=1,p=1,dist='t')
    except Exception as e:
        print('\nERROR > Could not fit GARCH Model: {e}')
        return None
    try:
        results=model.fit(disp='off')
    except Exception as e:
        # print(f'\nERROR > Issue Predicting Volatility: {e}')
        return None

    output_df[f'GARCH_volatility_{y_scaler}x'] = results.conditional_volatility
    output_df['GARCH_volatility_mult'] = output_df[f'GARCH_volatility_{y_scaler}x']/y_scaler

    return output_df



def run(feature_dataframe_array):
    print('>--------------------')
    print('\033[1mGARCHpipeline.py\033[0m')
    print('Fitting GARCH Model to each Dataframe')
    volatility_added_dataframes = []
    for i in range(len(feature_dataframe_array)):
        loading_bar(i,len(feature_dataframe_array))
        vol_df = generateVolatilityPrediction(feature_dataframe_array[i])
        if vol_df is not None:
            volatility_added_dataframes.append(vol_df)

    print('\nFinished Producing Volatility Predictions on the Feature data')
    return volatility_added_dataframes