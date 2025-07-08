import os
from datetime import date
from dotenv import load_dotenv
load_dotenv()


PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"



MODEL_FILE_NAME = "pipe.pkl"



TARGET_COLUMN = "sentiment"
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"



FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

#============================================AWS Configuration============================================================
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-southeast-2"
AWS_ACCESS_KEY_ID = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
AWS_SECRET_ACCESS_KEY = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)



"""Data Ingestion related constant start with DATA_INGESTION VAR NAME"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25



"""Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"



"""MODEL TRAINER related constant start with MODEL_TRAINER var name"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.65
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_N_ESTIMATORS=200
MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 7
MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6
MIN_SAMPLES_SPLIT_MAX_DEPTH: int = 10
MIN_SAMPLES_SPLIT_CRITERION: str = 'entropy'
MIN_SAMPLES_SPLIT_RANDOM_STATE: int = 101



"""MODEL Evaluation related constants"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "imdb15112001"
MODEL_PUSHER_S3_KEY = "model-registry"

