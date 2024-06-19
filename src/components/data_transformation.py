## It will be from data ingestion and out put will be transformed data and pipeline pickle file 
## Functions are handling missing valus,feature scaling,handling categories , encoding 
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from src.components.data_ingestion import  DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiated")

            df = pd.read_csv(os.path.join('artifacts','raw.csv'))
            target_column_name = 'Time_taken (min)'  
            df = df.drop(columns=target_column_name,axis=1)

            numerical_columns = df.select_dtypes(exclude='object').columns

            categorical_columns = df.select_dtypes(include='object').columns

            # Custom ranking variables

            Weather_conditions_Map=['Sunny','Sandstorms','Stormy','Windy','Cloudy','Fog']
            Road_traffic_density_Map=['Low','Medium','High','Jam']
            Type_of_vehicle_Map = ['scooter','electric_scooter','bicycle','motorcycle']
            Festival_Map=['No','Yes']
            City_Map=['Urban','Metropolitian','Semi-Urban']

            logging.info('Pipeline initiated')

            ## Numerical pipeline 
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ## Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[Weather_conditions_Map,Road_traffic_density_Map,Type_of_vehicle_Map,Festival_Map,City_Map])),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('Pipeline completed')

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
                # Reading train and test data 
                train_df = pd.read_csv(train_path)
                #print('Train df head',train_df.head().to_string())
                test_df =  pd.read_csv(test_path)
                #print('Train df head',train_df.head().to_string())

                logging.info('Reading of train and test data completed')
                logging.info(f'Train data head : \n{train_df.head().to_string()}')
                logging.info(f'Test data head : \n{test_df.head().to_string()}')

                logging.info('Obtaining preprocessing object')

                preprocessor_obj = self.get_data_transformation_object()  
                
                target_column_name = 'Time_taken (min)'  
              
                ## Independent / dependent features wrt training
                input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
                
                target_feature_train_df = train_df[target_column_name]
                
                ## Independent / dependent features wrt testing
                input_feature_test_df = test_df.drop(columns=target_column_name,axis=1)
                target_feature_test_df = test_df[target_column_name]

                ## Transforming using preprocessor object 
                ## The preprocessor object is initialised with the transformer name , the pipeline it should use and also the names of the respective columns that the pipelin eshould process 
                
                input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                
                input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

                logging.info("Applying preprocessing object on training and testing datasets.")

                ## Converting into np arrays makes it faster and c_ will concatenate both the arrays horizontally ---> |-|
                train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

                save_object(

                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessor_obj

                )
                logging.info('Preprocessor pickle file saved in artifacts')

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
        except Exception as e:
            logging.info('Exception occured at initiating data transformation')
            raise CustomException(e,sys)
        
# if __name__ == '__main__':

#     obj = DataIngestion()
#     train_path,test_path = obj.initiate_data_ingestion()

#     obj2 = DataTransformation()
#     train_arr,test_arr, object  = obj2.initiate_data_transformation(train_path,test_path)

#     obj3= ModelTrainer()
#     obj3.initiate_model_training(train_arr,test_arr)

