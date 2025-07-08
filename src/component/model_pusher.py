from src.cloud_storage.aws_storage import SimpleStorageService
from src.logger import configure_logger
logger = configure_logger("model_pusher")
from src.exception import MyException
from src.entity.artifact_entity import ModelPusherArtifact,ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import IMDBEstimator
import sys,os


class ModelPusher:

    def __init__(self,
                model_pusher_config:ModelPusherConfig,
                model_eval_artifact:ModelEvaluationArtifact):
        
        self.model_pusher_config = model_pusher_config
        self.model_eval_artifact = model_eval_artifact
        self.s3 = SimpleStorageService()
        self.imdb_estimator = IMDBEstimator(
            bucket_name =model_pusher_config.bucket_name,
            model_path = model_pusher_config.s3_model_key_path)

    

    def initiate_model_pusher(self):
        """This function push the model to S3"""
        try:
            logger.info("Pushing model to S3")
            self.imdb_estimator.save_model(from_file=self.model_eval_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name = self.model_pusher_config.bucket_name,
                s3_model_path = self.model_pusher_config.s3_model_key_path)

            logger.info("Model successfully pushed to AWS")
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e