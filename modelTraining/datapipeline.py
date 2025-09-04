import argparse
from dataPipelineAlgos import handleRawDataCollection as rawDataHandler
from dataPipelineAlgos import featureEngineering
from dataPipelineAlgos import GARCHpipeline
from dataPipelineAlgos import recursiveFeatureElimination
import os

# [] -> fix the model so that the scaling works better so i dont get convergence issues (rather have bad scaling than no data)
# [x] ->  ERROR > Could not perform RFE: float() argument must be a string or a real number, not 'Timestamp'
#       -> issue somewhere in the RFE (thinking it could be about the date - timestamp)
# [] -> make sure date is added to the output data csv files


def saveInputData(feature_dataframe_arr,ref_results):
    


    save_folder_name = 'dataPipelineOutputData'
    if not os.path.exists(save_folder_name):
        os.mkdir(save_folder_name)

    #first remove any existing files in the save location in case they're no longer included in the stock options but still exist resulting in unwanted data files
    for filename in os.listdir(save_folder_name):
        os.remove(f'{save_folder_name}/{filename}')
    
    for df in feature_dataframe_arr:
        ticker = df['Ticker'].iloc[0]
        save_df = df[ref_results]
        save_df.to_csv(os.path.join(save_folder_name,f'{ticker}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data Construction Pipeline for generating input data to financial ai models')
    parser.add_argument('--ticker_file',
                        required=True,
                        default='targetTickers.txt',
                        help='The file of tickers that will be used to decide which data will be downloaded')
    
    parser.add_argument('--target',
                        help='The target column for generating the data',
                        required=True,
                        default='Return')

    parser.add_argument('--min_days_of_data','-mdd',
                        help='The Minimum Number of days Worth of data a Stock Needs to have to be Downloaded',
                        default = 550)

    args = parser.parse_args()

    raw_data_frame_arr = rawDataHandler.run(args.ticker_file,args.min_days_of_data)
    feature_engineered_dataframe_arr = featureEngineering.run(raw_data_frame_arr,args.target)
    volatility_added_dataframe_arr = GARCHpipeline.run(feature_engineered_dataframe_arr)

    #now need to perform RFE in parallel
    rfe_results = recursiveFeatureElimination.run(volatility_added_dataframe_arr)
    if args.target not in rfe_results:
        rfe_results.append(args.target)

    if 'Date' not in rfe_results:
        rfe_results.append('Date')

    saveInputData(volatility_added_dataframe_arr,rfe_results)
    print('>--------------------')
    print('Data Pipeline Finished Running')
    print('>--------------------')

