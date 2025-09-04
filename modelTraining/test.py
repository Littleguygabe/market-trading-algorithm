import os
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    targ_dir = Path('dataPipelineOutputData')
    stock_dir = 'stockOptions.csv'

    #get the stocks listed in the stock dir
    stock_ops = pd.read_csv(stock_dir)
    stock_ops_list = stock_ops.values.flatten().tolist()
    targ_dir_tick_list = [file.stem for file in targ_dir.iterdir()] 

    stock_ops_set = set(stock_ops_list)
    not_in_stock_ops_file = [ticker for ticker in targ_dir_tick_list if ticker not in stock_ops_set]
    print(not_in_stock_ops_file)
