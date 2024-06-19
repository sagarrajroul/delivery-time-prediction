import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.exception import CustomException
from src.logger import logging
import pickle


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        # report={}
        # for i in range(len(models)):
        #     model=list(models.values())[i]
        #     #train model
        #     model.fit(X_train,y_train)

        #     #Predict Testing data
        #     y_test_pred=model.predict(X_test)

        #     #Get R2 scores for each model
        #     test_model_score=r2_score(y_test,y_test_pred)

        #     #report creation
        #     report[list(models.keys())[i]] = test_model_score

        # return report
        model.fit(X_train,y_train)

        #Predict Testing data
        y_test_pred=model.predict(X_test)

        #Get R2 scores for each model
        test_model_score=r2_score(y_test,y_test_pred)

        return test_model_score

    except Exception as e:

        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

## To load the pickle files        
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Exception occured in load_object function in utils')
        raise CustomException(e,sys)
