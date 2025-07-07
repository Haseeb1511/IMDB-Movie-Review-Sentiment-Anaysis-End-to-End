import sys
from src.exception import MyException
from src.logger import configure_logger
logger = configure_logger("training_pipeline")


from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_training import ModelTrainer

from src.entity.config_entity import DataIngestionConfig,DataTransformationConfig,ModelEvaluationConfig,ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact

class TrainingPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()


#=====================================DataIngestion===============================================
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """This method of TrainPipeline class is responsible for starting data ingestion component"""
        logger.info("-------------------------DataIngestion--------------------------------------")
        try:
            logger.info("Entered the start_data_ingestion method of TrainPipeline class")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Got the train_set and test_set")
            logger.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
#=====================================================DataTransformation=============================================

    def start_data_transformation(self,data_ingestion_artifact)->DataTransformationArtifact:
        try:

            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                    data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info("Exited the start data transformation part of train pipeline")
            return data_transformation_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
        

#===============================================ModelTrainer#####################################################

    def start_model_trainer(self,data_transformation_artifact):
        logger.info("Strt Model training")
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                            model_trainer_config=self.model_trainer_config
                                            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info("Model training finsihed")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e















    def run_pipeline(self ) -> None:
            """
            This method of TrainPipeline class is responsible for running complete pipeline
            """
            try:
                data_ingestion_artifact = self.start_data_ingestion()


                data_transformation_artifact = self.start_data_transformation(
                                data_ingestion_artifact=data_ingestion_artifact)
                
                model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

                # model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                #                                                         model_trainer_artifact=model_trainer_artifact)

                # if not model_evaluation_artifact.is_model_accepted:
                #     logger.info(f"Model not accepted.")
                    # return None
                
                # model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
                
            except Exception as e:
                raise MyException(e, sys)