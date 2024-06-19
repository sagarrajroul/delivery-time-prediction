#Whenever I give the input I should get the output
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            ## We write in this format instead of 'artifacts/filename.extension because this will run both on linux and windows
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occured in prediction pipeline")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                Delivery_person_Age:float,
                Delivery_person_Ratings:float,
                Weather_conditions:str,
                Road_traffic_density:str,
                Vehicle_condition:int,  
                Type_of_vehicle:str,
                multiple_deliveries:float,
                Festival:str,
                City:str,
                Ordered_Date_Year:int,  
                Ordered_Date_Month:int,  
                Ordered_Date_Day:int,  
                Time_OrderPicked_hours:int,  
                Time_OrderPicked_mins:int,  
                Time_Orderd_hours:int,  
                Time_Orderd_mins:int,
                Distance_covered:float
                ):
        
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition  = Vehicle_condition
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City
        self.Ordered_Date_Year  = Ordered_Date_Year
        self.Ordered_Date_Month  = Ordered_Date_Month
        self.Ordered_Date_Day  = Ordered_Date_Day
        self.Time_OrderPicked_hours  = Time_OrderPicked_hours
        self.Time_OrderPicked_mins  = Time_OrderPicked_mins
        self.Time_Orderd_hours  = Time_Orderd_hours
        self.Time_Orderd_mins = Time_Orderd_mins
        self.Distance_covered = Distance_covered

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Festival':[self.Festival],
                'City':[self.City],
                'Ordered_Date_Year':[self.Ordered_Date_Year],
                'Ordered_Date_Month':[self.Ordered_Date_Month],
                'Ordered_Date_Day':[self.Ordered_Date_Day],
                'Time_OrderPicked_hours':[self.Time_OrderPicked_hours],
                'Time_OrderPicked_mins':[self.Time_OrderPicked_mins],
                'Time_Orderd_hours':[self.Time_Orderd_hours],
                'Time_Orderd_mins':[self.Time_Orderd_mins],
                'Distance_covered':[self.Distance_covered]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)