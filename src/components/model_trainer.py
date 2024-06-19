# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from src.exception import CustomException
from src.logger import logging
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation  import DataTransformation

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'XGBoost_1': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),  # Modify hyperparameters
                'XGBoost_2': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5),  # Modify hyperparameters
                'LightGBM_1': LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=9, num_leaves=31),  # Modify hyperparameters
                'LightGBM_2': LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31),  # Modify hyperparameters
            }

            # models={
            #     'LinearRegression':LinearRegression(),
            #     'Lasso':Lasso(),
            #     'Ridge':Ridge(),
            #     'Elasticnet':ElasticNet()
            # }

            stacked_model = StackingRegressor(
            estimators= [(name, model) for name, model in models.items()],
            final_estimator= SVR()  # You can choose a different meta-estimator if needed
            )

            stacked_model.fit(X_train, y_train) # Train the stacked model
            y_test_pred = stacked_model.predict(X_test)  # Predict on the test data
            r2_score_result = r2_score(y_test, y_test_pred)  # Calculate the R2 score

            # Print or use r2_score_result as needed
            print("R2 Score:", r2_score_result)

            # model_report = evaluate_model(X_train,y_train,X_test,y_test,stacked_model)#models

            # print(model_report)

            # print('\n====================================================================================\n')
            # logging.info(f'Model Report : {model_report}')

            # # To get the best model score from the dictionary 
            # best_model_score = max(sorted(model_report.values()))
            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]

            # # Object of the best model
            # best_model = models[best_model_name]

            # print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            # print('\n====================================================================================\n')
            # logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= stacked_model   #best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
        
if __name__ == '__main__':

    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()

    obj2 = DataTransformation()
    train_arr,test_arr, object  = obj2.initiate_data_transformation(train_path,test_path)

    obj3= ModelTrainer()
    obj3.initiate_model_training(train_arr,test_arr)
