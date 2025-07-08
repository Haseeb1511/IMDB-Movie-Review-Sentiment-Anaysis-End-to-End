from src.entity.config_entity import PredictionConfig
from src.entity.s3_estimator import IMDBEstimator
from src.exception import MyException
import pandas as pd

import sys
import pandas as pd


class Data:
    def __init__(self, review):
        self.review = review

    def get_dataframe(self):
        try:
            return pd.DataFrame({"review": [self.review]})
        except Exception as e:
            raise MyException(e, sys) from e


class IMDBClassifier:
    def __init__(self, prediction_pipeline_config:PredictionConfig=PredictionConfig()):
        self.prediction_pipeline_config = prediction_pipeline_config


    def predict(self,dataframe):
        try:
            model = IMDBEstimator(
                bucket_name = self.prediction_pipeline_config.model_bucket_name,
                model_path = self.prediction_pipeline_config.model_file_path)
            result = model.predict(dataframe)
            return result
        except Exception as e:
            raise MyException(e, sys) from e
