import pandas as pd
import numpy as np
import os

### figure out which data values need normalising and normalise them

### use scatter graphs to plot a rough category of data against price (price Y axis, technical analytic X axis) -> could allow for classification???


def rolling_normalisation(df,col_names):
    #rolling window to stop from getting data leakage
    normalised_df = df.copy()
    for col in normalised_df.columns:
        if col in col_names:
            min_vals = normalised_df[col].expanding().min()
            max_vals = normalised_df[col].expanding().max()

            denominator = max_vals-min_vals
            denominator[denominator==0] = 1
            normalised_df[f'{col}_norm'] = (normalised_df[col]-min_vals)/denominator

    return normalised_df

def getMAs(rawdf):
    rawdf = rawdf.copy()
    # --- Calculate and assign all MAs to the DataFrame ---
    rawdf['SMA10'] = rawdf['Close'].rolling(window=10).mean()
    rawdf['SMA20'] = rawdf['Close'].rolling(window=20).mean()
    rawdf['SMA50'] = rawdf['Close'].rolling(window=50).mean()
    rawdf['EMA10'] = rawdf['Close'].ewm(span=10, adjust=False).mean()
    rawdf['EMA50'] = rawdf['Close'].ewm(span=50, adjust=False).mean()
    rawdf['EMA12'] = rawdf['Close'].ewm(span=12,adjust=False).mean()
    rawdf['EMA26'] = rawdf['Close'].ewm(span=26,adjust=False).mean()

    # --- Scale-free, relative MAs ---
    rawdf['SMA10_norm_rel'] = (rawdf['Close'] - rawdf['SMA10']) / rawdf['Close']
    rawdf['SMA20_norm_rel'] = (rawdf['Close'] - rawdf['SMA20']) / rawdf['Close']
    rawdf['SMA50_norm_rel'] = (rawdf['Close'] - rawdf['SMA50']) / rawdf['Close']
    rawdf['EMA10_norm_rel'] = (rawdf['Close'] - rawdf['EMA10']) / rawdf['Close']
    rawdf['EMA50_norm_rel'] = (rawdf['Close'] - rawdf['EMA50']) / rawdf['Close']
    rawdf['10to50_SMA_norm_diff'] = (rawdf['SMA10'] - rawdf['SMA50']) / rawdf['Close']
    rawdf['10to50_EMA_norm_diff'] = (rawdf['EMA10'] - rawdf['EMA50']) / rawdf['Close']

    return rawdf

def getMACDSIG(rawdf):
    rawdf = rawdf.copy()
    rawdf['MACDline'] = rawdf['EMA12'] - rawdf['EMA26']
    rawdf['SigLine'] = rawdf['MACDline'].ewm(span=9,adjust=False).mean()
    rawdf['normMACDSIGdif'] = (rawdf['MACDline'] - rawdf['SigLine']) * 100
    return rawdf

def getRSI(rawdf):
    rawdf = rawdf.copy()

    rawdf['DailyChange'] = rawdf['Close']-rawdf['Close'].shift(1)
    rawdf['Gain'] = np.maximum(rawdf['DailyChange'],0)
    rawdf['Loss'] = np.maximum(rawdf['DailyChange']*-1,0)

    rawdf['RelStren'] = (rawdf['Gain'].rolling(window=14).mean())/(rawdf['Loss'].rolling(window=14).mean())
    rawdf['RSI'] = 100 - (100/(1+rawdf['RelStren']))
    ### want to get how the rsi is changing, so want to calculate the difference between todays RSI and yesterdays RSI, then take the EMA of that
    rawdf['RSIprevDif'] = rawdf['RSI']-rawdf['RSI'].shift(1)
    return rawdf


def getATR(rawdf):
    rawdf = rawdf.copy()
    rawdf['trueRange'] = np.maximum(rawdf['High']-rawdf['Low'],np.maximum((rawdf['High']-(rawdf['Close'].shift(1))).abs(),(rawdf['Low']-(rawdf['Close'].shift(1))).abs()))
    rawdf['ATR'] = rawdf['trueRange'].rolling(window=14).mean()
    return rawdf

def getBB(rawdf):
    rawdf = rawdf.copy()

    rawdf['middleBand'] = rawdf['SMA20']
    rawdf['stdDev'] = rawdf['Close'].rolling(window=20).std()
    rawdf['upperBand'] = rawdf['middleBand']+2*rawdf['stdDev']
    rawdf['lowerBand'] = rawdf['middleBand']-2*rawdf['stdDev']

    #how close the close is to the lower band -> difference between close and low band value as a percentage of the band width
    rawdf['LBClsPctDif'] = (rawdf['Close']-rawdf['lowerBand'])/(rawdf['upperBand']-rawdf['lowerBand'])*100

    return rawdf

def getOBV(rawdf):
    rawdf = rawdf.copy()
    direction = np.sign(rawdf['Close'].diff()).fillna(0)
    obv = (rawdf['Volume'] * direction).cumsum()
    rawdf['OBV'] = obv
    rawdf['normalisedOBV'] = rawdf['OBV'] * 100
    rawdf['normOBVEMA25'] = rawdf['normalisedOBV'].ewm(span=25, adjust=False).mean()
    return rawdf

def getReturns(rawdf):
    rawdf=rawdf.copy()
    rawdf['Return'] = np.log(rawdf['Close']/rawdf['Close'].shift(1))

    return rawdf


def getFeatureData(rawdf):
    rawdf = getMAs(rawdf)
    rawdf = getMACDSIG(rawdf)
    rawdf = getRSI(rawdf)
    rawdf = getATR(rawdf)
    rawdf = getBB(rawdf)
    rawdf = getOBV(rawdf)
    rawdf = getReturns(rawdf)

    features_to_normalise = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
    'SMA10', 'SMA20', 'SMA50', 'EMA10', 'EMA50', 'EMA12', 'EMA26',
    'MACDline', 'SigLine', 'DailyChange', 'Gain', 'Loss',
    'trueRange', 'ATR', 'middleBand', 'stdDev', 'upperBand',
    'lowerBand', 'OBV'
]

    feature_df = rolling_normalisation(rawdf,features_to_normalise) 


    final_feature_columns = [
    'Date',
    'Ticker',
    # Inherently scaled or ratio-based features
    'RSI',
    'Return',
    'RelStren',
    'RSIprevDif',
    'LBClsPctDif',
    'Stock Splits', # Event flag, not scale-dependent

    # Pre-engineered relative/difference features
    'SMA10_norm_rel', 'SMA20_norm_rel', 'SMA50_norm_rel',
    'EMA10_norm_rel', 'EMA50_norm_rel', '10to50_SMA_norm_diff',
    '10to50_EMA_norm_diff', 'normMACDSIGdif',

    # Normalized versions of core price/volume data
    'Open_norm', 'High_norm', 'Low_norm', 'Close_norm', 'Volume_norm',

    # Normalized versions of moving averages
    'SMA10_norm', 'SMA20_norm', 'SMA50_norm', 'EMA10_norm', 'EMA50_norm',
    'EMA12_norm', 'EMA26_norm',

    # Normalized versions of other indicators
    'MACDline_norm', 'SigLine_norm', 'Gain_norm', 'Loss_norm',
    'trueRange_norm', 'ATR_norm', 'middleBand_norm', 'stdDev_norm',
    'upperBand_norm', 'lowerBand_norm',

    # Normalized versions of OBV
    'OBV_norm',
    'normalisedOBV',  # Assuming this is a different normalization method
    'normOBVEMA25'
]

    existing_final_cols = [col for col in final_feature_columns if col in feature_df.columns]

    final_df = feature_df[existing_final_cols].dropna().reset_index(drop=True)
    return final_df
 
def run(raw_dataframe,verbose=0):
    if verbose>-1:
        print('>--------------------')
        print('\033[1mRunning featureEngineering.py\033[0m')

    output_df = getFeatureData(raw_dataframe)

    if verbose>-1:
        print('\nNon-Volatility Features Created')

    return output_df
