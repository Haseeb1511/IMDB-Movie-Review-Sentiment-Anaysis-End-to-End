from src.entity.config_entity import DataTransformationConfig,DataIngestionConfig,SCHEMA_FILE_PATH
from src.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact

from src.logger import configure_logger
logger = configure_logger("DataTransformation")
from src.exception import MyException
from src.utils.main import save_object, save_numpy_array_data, read_yaml_file
import sys,os
import pandas as pd
from sklearn.pipeline import FunctionTransformer,Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.constant import TARGET_COLUMN
import re
import numpy as np


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download("wordnet")
wn = WordNetLemmatizer()
sw = set(stopwords.words("english"))




class DataTransformation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_transformation_config:DataTransformationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    @staticmethod
    def read_data(file_path):
        return pd.read_csv(file_path)
    

    def clean_text(self,text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove links (http, https, www)
        text = re.sub(r'[^a-zA-Z\s]', " ", text)
        text = re.sub(r'\s+', ' ', text).strip() 
        text = text.lower().split()
        text = " ".join(wn.lemmatize(word) for word in text if word not in sw )
        return text
    

    def response_map(self,data):
        data[TARGET_COLUMN] = data[TARGET_COLUMN].map({"positive":1,"negative":0})
        return data
    
    #FunctionTransformer(lambda x: np.array([self.clean_text(text) for text in x.ravel()]), validate=False)
    def preprocess_pipeline(self):
        text_pipline = Pipeline(steps=[
        ("clean_text", FunctionTransformer(lambda x: x.apply(self.clean_text))),
        ("tfidf", TfidfVectorizer(max_features=100))])

        column = ColumnTransformer(transformers=[
            ("text_pipeline", text_pipline, "review")])
        return column
        
    
    def initiate_data_transformation(self):
        try:
            train_data = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_data = self.read_data(self.data_ingestion_artifact.test_file_path)
            logger.info("train and test data loaded")

            train_data = self.response_map(train_data)
            test_data = self.response_map(test_data)
            logger.info("label encoded")

            input_feature_train_df = train_data.drop(TARGET_COLUMN,axis=1)
            target_feature_train_df = train_data[TARGET_COLUMN]
            logger.info(f"Shape of input_feature_train_df: {input_feature_train_df.shape}")

            input_feature_test_df = test_data.drop(TARGET_COLUMN,axis=1)
            target_feature_test_df = test_data[TARGET_COLUMN]

            preprocessed = self.preprocess_pipeline()         # transformation pipline
            logger.info("Preprocessed object loaded")

            #Transform train and test data(input)
            input_feature_train_array_sparse = preprocessed.fit_transform(input_feature_train_df)
            input_feature_test_array_sparse = preprocessed.transform(input_feature_test_df)
            logger.info("Train and test data transfoed using Transform object")
            logger.info(f"Shape of input_feature_train_array: {input_feature_train_array_sparse.shape}")


             #these two lines are exactly correct for your use case — because TfidfVectorizer always returns a sparse matrix by default, and np.c_[] needs dense NumPy arrays.
             # Explicitly convert sparse arrays to dense NumPy arrays=====> aas tfid give (750,100)  while input is (750,1)
            input_feature_train_array = input_feature_train_array_sparse.toarray()
            input_feature_test_array = input_feature_test_array_sparse.toarray()
            
            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df).reshape(-1,1)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df).reshape(-1,1)]
            logger.info("Train and test array concatenated to save them")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessed)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
        
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys) from e


