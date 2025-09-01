import xgboost as xgb
import os
import sys
def loadModel(model_path):
    try:
        loaded_bst = xgb.Booster()
        loaded_bst.load_model(os.path.join(model_path,'model.json'))

    except Exception as e:
        print(f'ERROR > Could not Load Model from: {model_path}')
        print(f'REASON: {e}')
        sys.exit(1)

    return loaded_bst


def generatePrediction(input_data,model):
    dnew = xgb.DMatrix(input_data.tail(1))
    prediction = model.predict(dnew)
    return prediction

def run(model_path,feature_data,verbose):
    if verbose>-1:
        print('>--------------------')
        print('\033[1mRunning xgboostPrediction.py\033[0m')
    model = loadModel(model_path)
    prediction = generatePrediction(feature_data,model)[0]
    if verbose>-1:
        print(f'Predicted Return > {round(prediction*100,5)}%')
    return prediction