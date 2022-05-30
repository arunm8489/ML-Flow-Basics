from utils.common import read_config
from utils.sagemaker_integration import query
import pandas as pd
import mlflow


data = pd.read_json('{"fixed acidity":{"0":7.4},"volatile acidity":{"0":0.7},"citric acid":{"0":0},"residual sugar":{'
                    '"0":1.9},"chlorides":{"0":0.076},"free sulfur dioxide":{"0":11},"total sulfur dioxide":{"0":34},'
                    '"density":{"0":0.9978},"pH":{"0":3.51},"sulphates":{"0":0.56},"alcohol":{"0":9.4}}')

config = read_config('config.yaml')

if __name__ == "__main__":
    print('Input: ')
    print('')
    input_json = data.to_json(orient='split')
    print(input_json)
    print('--'*20)
    print('')
    print('Output: ')
    Response = query(input_json,config)
    print(f"Predictions From Model EndPoint : {Response}")

