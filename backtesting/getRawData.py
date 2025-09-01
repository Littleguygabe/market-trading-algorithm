import yfinance as yf
import pandas as pd

ticker = 'MSFT'
data = yf.Ticker(ticker).history(period='2000d',interval='1d')
data.to_csv(f'{ticker}.csv')