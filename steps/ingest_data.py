import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from data_path
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): _description_
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Reads data from the specified Data Path
        
        Returns:
            df: dataframe containing the ingested data
        """
        logging.info(f"Reading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        return df
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingest data from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the ingested data.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise e
    
    