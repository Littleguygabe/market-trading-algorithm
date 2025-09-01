from arch import arch_model
import os
import sys
import pandas as pd
import numpy as np
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
        print(f'\nERROR > Issue Predicting Volatility: {e}')
        return None

    # volatility for input features - needed for the next day's return prediction
    output_df[f'GARCH_volatility_{y_scaler}x'] = results.conditional_volatility
    output_df['GARCH_volatility_mult'] = output_df[f'GARCH_volatility_{y_scaler}x']/y_scaler

    # predicted volatility for the next day - used alongside the return prediction
    forecast = results.forecast(horizon=1)
    pred_variance = forecast.variance.iloc[-1,0]
    pred_volatility = np.sqrt(pred_variance)    

    return output_df,pred_volatility



def run(feature_df,verbose):
    if verbose>-1:
        print('>--------------------')
        print('\033[1mRunning GARCHpipeline.py\033[0m')
        print('Fitting GARCH Model to each Dataframe')


    vol_df,pred_volatility = generateVolatilityPrediction(feature_df)
    if verbose>-1:
        print('\nFinished Producing Volatility Predictions on the Feature data')
    return vol_df, pred_volatility