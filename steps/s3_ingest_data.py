import boto3
import pandas as pd
import io
import logging # Import the logging module
import os # For better credential handling (optional)

# --- Configure logging ---
# You can customize this logging configuration based on your needs.
# For example, write to a file, set different levels, etc.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger specific to this module

class S3CSVReader:
    """
    A class to read CSV files directly from an AWS S3 bucket into a pandas DataFrame.
    Includes logging for better traceability.
    """

    def __init__(self, bucket_name: str, region_name: str = 'us-east-1',
                 aws_access_key_id: str = None, aws_secret_access_key: str = None):
        """
        Initializes the S3CSVReader with S3 bucket details and AWS credentials.

        Args:
            bucket_name (str): The name of the S3 bucket.
            region_name (str): The AWS region of the S3 bucket (default: 'us-east-1').
            aws_access_key_id (str, optional): AWS Access Key ID. If None, boto3 will
                                               look for credentials in environment variables,
                                               shared credential file (~/.aws/credentials),
                                               or IAM roles.
            aws_secret_access_key (str, optional): AWS Secret Access Key. Similar to above.
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_client = self._initialize_s3_client()
        logger.info(f"S3CSVReader initialized for bucket '{self.bucket_name}' in region '{self.region_name}'.")

    def _initialize_s3_client(self):
        """
        Initializes and returns an S3 client.
        Prioritizes explicit keys, then environment variables, then IAM roles/CLI config.
        """
        if self.aws_access_key_id and self.aws_secret_access_key:
            logger.info("Initializing S3 client with explicit AWS access keys.")
            return boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            )
        else:
            logger.info("No explicit AWS keys provided. boto3 will attempt to find credentials "
                        "from environment variables, AWS config, or IAM roles.")
            return boto3.client('s3', region_name=self.region_name)

    def read_csv(self, s3_key: str, encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
        """
        Reads a CSV file from the specified S3 key directly into a pandas DataFrame.

        Args:
            s3_key (str): The full path to the CSV file within the S3 bucket
                          (e.g., 'data/my_file.csv').
            encoding (str): The encoding of the CSV file (default: 'utf-8').
                            Change this if you encounter UnicodeDecodeError.
            **kwargs: Additional keyword arguments to pass to pandas.read_csv().

        Returns:
            pd.DataFrame: A pandas DataFrame containing the CSV data.

        Raises:
            FileNotFoundError: If the S3 object is not found.
            PermissionError: If there are insufficient permissions to access the S3 object.
            UnicodeDecodeError: If the specified encoding is incorrect for the file content.
            pd.errors.EmptyDataError: If the CSV file is empty.
            Exception: For other unexpected errors.
        """
        logger.info(f"Attempting to read '{s3_key}' from bucket '{self.bucket_name}' with encoding '{encoding}'.")
        try:
            logger.debug(f"Calling get_object for Bucket='{self.bucket_name}', Key='{s3_key}'.")
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully fetched object '{s3_key}'.")

            body = response['Body']
            csv_string = body.read().decode(encoding)
            logger.debug("CSV content decoded from S3 object.")

            df = pd.read_csv(io.StringIO(csv_string), **kwargs)
            logger.info(f"Successfully loaded '{s3_key}' into DataFrame. Shape: {df.shape}")
            return df

        except self.s3_client.exceptions.NoSuchKey:
            logger.error(f"S3 object '{s3_key}' not found in bucket '{self.bucket_name}'.")
            raise FileNotFoundError(f"S3 object '{s3_key}' not found in bucket '{self.bucket_name}'.")
        
        except self.s3_client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == 'AccessDenied':
                logger.critical(f"Access denied to S3 object '{s3_key}' in bucket '{self.bucket_name}'. "
                                f"Check your IAM permissions. Error details: {e}")
                raise PermissionError(f"Access denied to S3 object '{s3_key}' in bucket '{self.bucket_name}'. "
                                      "Check your IAM permissions.")
            else:
                logger.error(f"S3 Client Error for '{s3_key}': {e}")
                raise Exception(f"S3 Client Error for '{s3_key}': {e}")
            
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode CSV from '{s3_key}' with encoding '{encoding}'. "
                         f"Error: {e}. Try a different encoding (e.g., 'latin-1', 'cp1252').")
            raise UnicodeDecodeError(f"Failed to decode CSV from '{s3_key}' with encoding '{encoding}'. "
                                     f"Error: {e}. Try a different encoding (e.g., 'latin-1', 'cp1252').")
            
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file '{s3_key}' is empty.")
            return pd.DataFrame() # Return empty DataFrame for empty CSVs
        
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing '{s3_key}'.") # exception logs traceback
            raise Exception(f"An unexpected error occurred while processing '{s3_key}': {e}")