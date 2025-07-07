from sklearn.ensemble import RandomForestClassifier
from src.logger import configure_logger
logger = configure_logger("ModelTrainning")
from src.exception import MyException

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact,ClassificationMetricArtifact

from src.utils.main import load_numpy_array_data, load_object, save_object
from src.constant import TARGET_COLUMN
import numpy as np
import pandas as pd
import os,sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
from sklearn.pipeline import Pipeline


class ModelTrainer:

    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    
    def get_model_object(self,train:np.ndarray,test:np.ndarray,pipe):
        try:
            x_train,y_train,x_test,y_test = train[:,:-1],train[:,-1],test[:,:-1],test[:,-1]

            pipe.fit(x_train,y_train)

            y_pred = pipe.predict(x_test)

            accuracy = accuracy_score(y_test,y_pred)
            percision  = precision_score(y_test, y_pred, average='binary')
            f1score = f1_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            logger.info(f"Model training finished with accuracy of : {accuracy}")
            metric_artifact = ClassificationMetricArtifact(
                    f1_score=f1score,
                    precision_score=percision,
                    recall_score=recall,
                    accuracy=accuracy
                )
            return pipe,metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
    

    def initiate_model_trainer(self) ->ModelTrainerArtifact:
        try:
            train_array =load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_array =load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            preprocess_object = load_object(self.data_transformation_artifact.transformed_object_file_path)
            
            model_pipeline = Pipeline([
                ("preprocessing", preprocess_object),
                ("classifier", RandomForestClassifier(
                                    n_estimators=self.model_trainer_config._n_estimators,
                                    max_depth=self.model_trainer_config._max_depth,
                                    min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                                    min_samples_split=self.model_trainer_config._min_samples_split,
                                    criterion=self.model_trainer_config._criterion,
                                    random_state=self.model_trainer_config._random_state))])
                                
            
            trained_model,metric_artifact = self.get_model_object(train=train_array,test=test_array,pipe=model_pipeline)

            y_train = train_array[:, -1]
            y_pred = trained_model.predict(train_array[:, :-1])

            if accuracy_score(y_train,y_pred) < self.model_trainer_config.expected_accuracy:
                raise Exception("No model found with score above the base score")
            
            #Save both preprocessing object and trained model to use it for Inference
            save_object(self.model_trainer_config.trained_model_file_path, model_pipeline)

            model_trainer_artifact = ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    metric_artifact=metric_artifact
                )
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e