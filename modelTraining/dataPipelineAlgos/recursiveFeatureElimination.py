import multiprocessing
from dataPipelineAlgos.loading_bar import loading_bar
from sklearn.feature_selection import RFE
import xgboost as xgb
import pandas as pd
from tqdm import tqdm
import numpy as np


def performRFE(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    X = df.drop(columns=['target','Ticker','Date'])
    y = df['target']


    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs = 1,
    )

    rfe = RFE(estimator=model,n_features_to_select=10)
    try:
        rfe.fit(X, y)
        ranks = pd.Series(rfe.ranking_, index=X.columns, name='ranking')
        return ranks.to_frame().T
    except Exception as e:
        print(f'\nERROR > Could not perform RFE: {e}\n')
        return None


def run(volatility_added_df_arr):
    n_features_to_keep = 15
    print('>--------------------')
    print('\033[1mRunning recursiveFeatureElimination.py\033[0m')

    NUM_JOBS = multiprocessing.cpu_count()
    tasks = volatility_added_df_arr
    print('Starting Multi-Processing')
    with multiprocessing.Pool(processes=NUM_JOBS) as pool:
        results = list(tqdm(pool.imap_unordered(performRFE, tasks), total=len(tasks)))

    successful_results = [res for res in results if res is not None]

    output_df = pd.concat(successful_results,ignore_index=True)
    total_ranks = output_df.sum(axis=0)
    most_influential_features = total_ranks.sort_values(ascending=True)
    print('RFE Finished running')
    final_feature_df =  most_influential_features[:n_features_to_keep]
    return final_feature_df.index.tolist()