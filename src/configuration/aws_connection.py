import boto3
import os
from src.constant import AWS_ACCESS_KEY_ID_ENV_KEY,AWS_SECRET_ACCESS_KEY_ENV_KEY,REGION_NAME
from dotenv import load_dotenv
load_dotenv()
import botocore

config = botocore.config.Config(
    connect_timeout=60,
    read_timeout=120,
    retries={'max_attempts': 5}
)

class S3Client:
    s3_client = None
    s3_resource = None

    def __init__(self,region_name=REGION_NAME):
        if S3Client.s3_client ==None or S3Client.s3_resource==None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
            if __access_key_id is None:
                raise Exception(f"Enviroment variable is not set{AWS_ACCESS_KEY_ID_ENV_KEY}")
            if __secret_access_key is None:
                raise Exception(f"Enviroment variable is not set{AWS_SECRET_ACCESS_KEY_ENV_KEY}")
            
            S3Client.s3_resource = boto3.resource('s3',
                                            aws_access_key_id=__access_key_id,
                                            aws_secret_access_key=__secret_access_key,
                                            region_name=region_name,config=config
                                            )
            S3Client.s3_client = boto3.client('s3',
                                        aws_access_key_id=__access_key_id,
                                        aws_secret_access_key=__secret_access_key,
                                        region_name=region_name,config=config
                                        )
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client

