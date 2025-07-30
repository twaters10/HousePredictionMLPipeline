import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Union, Tuple
from io import StringIO
import boto3
from config.access_keys import *

def clean_data(df: pd.DataFrame) -> Annotated[pd.DataFrame, "processed_data"]:
    """Cleans the input data frame.
    Args:
        df (pd.DataFrame): input data frame to be cleaned.

    Raises:
        e: error in processing data cleaning.

    Returns:
        pd.DataFrame: Processed data frame after cleaning.
    """
    try:
        logging.info("Starting data cleaning process...")
        # Process Data Cleaning
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        logging.info("Data preprocessing completed successfully.")
        return processed_data
    except Exception as e:
        logging.error(f"Error in data cleaning step: {e}")
        raise e

def split_data(df: pd.DataFrame) -> Tuple[
                                    Annotated[pd.DataFrame, "X_train"],
                                    Annotated[pd.DataFrame, "X_val"],
                                    Annotated[pd.Series, "y_train"],
                                    Annotated[pd.Series, "y_val"]
                                    ]:
    """Divides it into training and validation sets.

    Args:
        df (pd.DataFrame): cleaned data frame to be split.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.

    Raises:
        e: error in processing data splitting.
    """
    try:
        # Process Data Splitting
        logging.info("Dividing data into training and validation sets...")
        divide_strategy = DataSplitStrategy()
        data_split = DataCleaning(df, divide_strategy)
        X_train, X_val, y_train, y_val = data_split.handle_data()
        logging.info("Data splitting completed successfully.")
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        logging.error(f"Error in data cleaning step: {e}")
        raise e
    
def load_processed_data_to_s3(df: pd.DataFrame, bucket_name: str, csvfilename: str) -> None:
    """Loads the processed data to S3 bucket.

    Args:
        df (pd.DataFrame): Processed data frame to be uploaded.
        bucket_name (str): Name of the S3 bucket.
        csvfilename (str): Name of the CSV file to be created in S3.
        
    Raises:
        e: error in uploading processed data to S3.
    """
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client = boto3.client('s3')
        bucket_name = bucket_name
        object_key = csvfilename
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())
        logging.info(f"Processed data saved to S3 bucket {bucket_name} at key {object_key}.")
    except Exception as e:
        logging.error(f"Error uploading processed data to S3: {e}")
        raise e