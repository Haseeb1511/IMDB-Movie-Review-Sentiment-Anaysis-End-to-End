from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score
from src.exception import MyException
from src.logger import configure_logger
logger = configure_logger("model_evaluation")
from src.constant import TARGET_COLUMN,SCHEMA_FILE_PATH
from src.utils.main import load_object
import os,sys
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.main import read_yaml_file
from src.entity.s3_estimator import IMDBEstimator

from dataclasses import dataclass
@dataclass
class EvaluateModelResponse:
    trained_model_accuracy: float
    best_model_accuracy_score: float
    is_model_accepted: bool
    difference: float



class ModelEvaluation:
   
    def __init__(self,model_eval_config:ModelEvaluationConfig,data_ingestion_artifact:DataIngestionArtifact,
        model_trainer_artifact:ModelTrainerArtifact):

        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    
    def get_best_model(self):
        try:
            """This function help us in getting best model for S3"""
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            estimator  = IMDBEstimator(bucket_name=bucket_name,
                                        model_path=model_path)
            if estimator.is_model_present(model_path=model_path):
                return estimator
            return None
        except Exception as e:
            raise MyException(e,sys) from e
       
        
    

    def evaluate_model(self):
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x = test_df.drop(TARGET_COLUMN,axis=1)
            y = test_df[TARGET_COLUMN]

            trained_model = load_object(file_path = self.model_trainer_artifact.trained_model_file_path)

            train_model_accuracy = self.model_trainer_artifact.metric_artifact.accuracy

            best_model_accuracy = None
            best_model = self.get_best_model()

            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_accuracy = accuracy_score(y,y_hat_best_model)
                logger.info(f"Accuracy_Score-Production Model: {best_model_accuracy}, Accuracy_Score-New Trained Model: {train_model_accuracy}")
            else:
                best_model_accuracy=0
            temp_model_best_score = best_model_accuracy
            
            return EvaluateModelResponse(
                trained_model_accuracy = train_model_accuracy,
                best_model_accuracy_score = best_model_accuracy,
                is_model_accepted = train_model_accuracy > best_model_accuracy,
                difference = train_model_accuracy-temp_model_best_score)
        
        except Exception as e:
            raise MyException(e,sys) from e
        


    
    def initiate_model_evaluation(self):
        evaluat_model = self.evaluate_model()

        return  ModelEvaluationArtifact(
            is_model_accepted = evaluat_model.is_model_accepted,
            changed_accuracy=evaluat_model.difference,
            s3_model_path = self.model_eval_config.s3_model_key_path,
            trained_model_path= self.model_trainer_artifact.trained_model_file_path)



