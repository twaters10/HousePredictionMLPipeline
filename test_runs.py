from steps.s3_ingest_data import *
from config.access_keys import *
import boto3
import pandas as pd
import io
import logging # Import the logging module
import os # For better credential handling (optional)
if __name__ == "__main__":
    logging.info("Starting S3CSVReader...")
    reader = S3CSVReader(bucket_name=S3_BUCKET_NAME, region_name=AWS_REGION, 
                         aws_access_key_id=S3_AWS_ACCESS_KEY_ID, aws_secret_access_key=S3_AWS_SECRET_ACCESS_KEY)
    
    df = reader.read_csv(s3_key=S3_KEY, encoding='utf-8')
    print(df.head(5))
    
    
    
    