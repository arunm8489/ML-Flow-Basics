import os
import warnings
import sys
import pandas as pd
import numpy as np 
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import mlflow
import sys
import warnings


from sklearn.ensemble import RandomForestRegressor
def eval_metric(actual,pred):
    #compute relevant metrics
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2  

def load_data(data_path ='data/winequality-red.csv'):
    data = pd.read_csv(data_path)
    X = data.drop(["quality"], axis=1)
    y = data['quality']
    X_train,X_test,y_train, y_test =  train_test_split(X,y,test_size=0.25,random_state=42)
    return X_train,y_train,X_test,y_test


def train_model(no_estimators,max_depth,run_name="best"):
    np.random.seed(40)
    X_train,y_train,X_test,y_test = load_data()
    with mlflow.start_run(run_name=run_name) as run:
        print(f'Run id: {run.info.run_uuid}')
        print(f'Run name: {run_name}')
        print(f'Exp id: {run.info.experiment_id}')
        model = RandomForestRegressor(n_estimators=no_estimators,max_depth=max_depth)
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = eval_metric(y_train,y_train_pred)
        test_metrics = eval_metric(y_test,y_test_pred) 

        mlflow.log_params({"n_estimators":no_estimators,"max_depth":max_depth})
        mlflow.log_metrics({"train_rmse":train_metrics[0],"train_mae": train_metrics[1], "r2": train_metrics[2]})
        mlflow.log_metrics({"test_rmse":test_metrics[0],"test_mae": test_metrics[1], "r2": test_metrics[2]})
        mlflow.log_artifact(data_path)
        print("Save to: {}".format(mlflow.get_artifact_uri()))
        mlflow.sklearn.log_model(model, "model")
        
        # Print out metrics
        print(f"Random Forest model:")  
        print(f"n_estimators:{no_estimators}, max_depth:{max_depth}")
        print("  Train RMSE: %s" % train_metrics[0])
        print("  Train MAE: %s" % train_metrics[1])
        print("  Train R2: %s" % train_metrics[2])
        print('--'*50)
        print("  Test RMSE: %s" % test_metrics[0])
        print("  Test MAE: %s" % test_metrics[1])
        print("  Test R2: %s" % test_metrics[2])
        mlflow.end_run()


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    data_path = 'data/winequality-red.csv'
    X_train,X_test,y_train, y_test =  load_data(data_path)

    no_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    exp_name = sys.argv[3]
    train_model(no_estimators,max_depth,exp_name)
    