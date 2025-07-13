from src.configuration.aws_connection import S3Client
from io import StringIO
from src.constant import MODEL_BUCKET_NAME
import pandas as pd

s3_client = S3Client()

def fetch_file_from_s3(file_key):
        """Fetches a CSV file from the S3 bucket and returns it as a Pandas DataFrame."""
        try:
            obj = s3_client.s3_client.get_object(Bucket=MODEL_BUCKET_NAME, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            return df
        
        except Exception as e:
            return None

