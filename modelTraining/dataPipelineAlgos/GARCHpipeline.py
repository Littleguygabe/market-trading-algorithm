from arch import arch_model
from tqdm import tqdm
import numpy as np
import warnings
from arch.utility.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

def generateVolatilityPrediction(df):
    output_df = df.copy()
    returns = output_df['Return'].dropna()

    if np.isclose(returns.std(), 0):
        print("\nSkipping stock with zero volatility.")
        return None

    # dynamic model selection

    models_to_try = [
        {'model':'EGARCH','p':1,'o':1,'q':1},
        {'model':'GARCH','p':1,'q':1},
        {'model':'ARCH','p':1},
        
    ]
    scaler = 100
    results = None
    for config in models_to_try:
        try:    
            model_config = config.copy()
            vol_model_name = model_config.pop('model')

            model = arch_model(returns*scaler,vol_model_name,**model_config,dist='t')
            fit_results = model.fit(disp='off')
            if fit_results.convergence_flag == 0:
                results = fit_results
                break
            
        except Exception:
            continue

    if results is None:
        return None

    output_df['GARCH_volatility'] = results.conditional_volatility / scaler

    return output_df



def run(feature_dataframe_array):
    print('>--------------------')
    print('\033[1mGARCHpipeline.py\033[0m')
    print('Fitting GARCH Model to each Dataframe')
    volatility_added_dataframes = []
    for i in tqdm(range(len(feature_dataframe_array))):
        vol_df = generateVolatilityPrediction(feature_dataframe_array[i])
        if vol_df is not None:
            volatility_added_dataframes.append(vol_df)

    print('\nFinished Producing Volatility Predictions on the Feature data')
    return volatility_added_dataframes