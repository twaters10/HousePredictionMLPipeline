from steps.s3_ingest_data import *
from steps.clean_data import *
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
    # Read the CSV file from S3
    df = reader.read_csv(s3_key=S3_KEY, encoding='utf-8')
    
    # Clean, transform, and split the data.
    processed_df = clean_data(df)
    X_train, y_train, X_val, y_val = split_data(processed_df)
    
    # Load the processed data back to S3
    load_processed_data_to_s3(processed_df, bucket_name=S3_BUCKET_NAME, csvfilename='processed_house_prices.csv')
    
    
    
    