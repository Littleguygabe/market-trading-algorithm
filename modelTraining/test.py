import yfinance as yf
import pandas as pd
import multiprocessing


def flatten(dataframe,ticker):
    

if __name__ == '__main__':
    df_ticker_list = pd.read_csv('stockOptions.csv')
    tickers_list = df_ticker_list['Ticker'].to_numpy().tolist()
    tickers = yf.Tickers(tickers_list)
    data = tickers.download(period='100d',interval='1d')

