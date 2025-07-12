from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import configure_logger
logger = configure_logger("s3_estimator")
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame
import pickle
import pandas as pd



class IMDBEstimator:

    def __init__(self,bucket_name,model_path):
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model:MyModel=None

    def is_model_present(self,model_path):
        """This function check if model is present in s3 or not"""
        return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
    

    def load_model(self,)->MyModel:
        """This function load the model from S3"""
        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)
    
    
    def save_model(self,from_file,remove:bool=False)->None:
        """This function save the model to S3"""
        self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        
    def predict(self,input):
        """This function is used for Prediction using the model from S3"""
        if self.loaded_model is None:
            self.loaded_model = self.load_model()
        return self.loaded_model.predict(input)
    