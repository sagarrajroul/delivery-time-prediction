import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # explore dataclass

#from src.components.data_transformation import DataTransformation

## Initialize the Data Ingetion Configuration (create data for training and testing)
## This module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str  = os.path.join('artifacts','raw.csv')

## Create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion starts - Creation of train and test data')
        try:
            df =pd.read_csv(os.path.join('notebooks/data','clean_finalTrain.csv'))
            logging.info('Dataset was read to pandas')

            columns = df.columns.tolist()
            columns.remove('Time_taken (min)')
            columns.append('Time_taken (min)')
            df= df[columns]

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Splitting into train and test')
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)
            logging.info(train_set.head())
            logging.info(test_set.head())

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)    
            test_set.to_csv(self.ingestion_config.test_data_path,index =False, header = True)

            logging.info('Data ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info('Exception found at Data Ingestion stage')
            raise CustomException(e,sys)

# if __name__ == '__main__':
#     obj= DataIngestion()
#     train_data_path,test_data_path = obj.initiate_data_ingestion()
#     print(train_data_path,'     ##################    ',test_data_path)