import os
import yfinance as yf
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd

TICKER = 'MSFT'

def getRawData():
    ticker_obj = yf.Ticker(TICKER)
    data = ticker_obj.history(period='2000d',interval='1d')
    return data

def generateIndicators(raw_data):
    df = raw_data.copy()
    df['Indicator'] = np.random.randint(-1,2,len(df))
    return df

def generateOrderValue(df):
    df_copy = df.copy()
    df_copy['OrderVal'] = np.random.randint(0,501,len(df))
    return df_copy

def saveData(output):
    save_path = Path(__file__).parent.parent
    path = save_path/'backTestingData'
    output.to_csv(os.path.join(path,f'{TICKER}.csv'))





if __name__ == '__main__':
    # get the ticker data
    raw_data = getRawData()
    full_data = generateIndicators(raw_data)
    full_data = generateOrderValue(full_data)
    output = full_data[['Close','Indicator','OrderVal']].copy()
    
    saveData(output)