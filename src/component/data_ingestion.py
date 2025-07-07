from src.logger import configure_logger
logger = configure_logger("data_ingestion")
from src.exception import MyException

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
from sklearn.model_selection import train_test_split

import os,sys

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def load_data(self,data_path):
        data = pd.read_csv(data_path,nrows=1000)
        return data
    
    def split_data(self,dataframe:pd.DataFrame):
        train_data,test_data = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)

        #make dir
        train_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
        test_dir = os.path.dirname(self.data_ingestion_config.testing_file_path)
        os.makedirs(train_dir,exist_ok=True)
        os.makedirs(test_dir,exist_ok=True)
        
        # save the train and test data
        train_data.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
        test_data.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)


    def initiate_data_ingestion(self)->DataIngestionArtifact:
        data = self.load_data("data/IMDB.csv")
        self.split_data(data)

        return DataIngestionArtifact(
            trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path
        )

    

        
