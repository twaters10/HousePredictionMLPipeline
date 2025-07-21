import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Union, Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
                                        Annotated[pd.DataFrame, "X_train"],
                                        Annotated[pd.DataFrame, "X_val"],
                                        Annotated[pd.Series, "y_train"],
                                        Annotated[pd.Series, "y_val"]
                                    ]:
    """Cleans the data and divides it into training and validation sets.

    Args:
        df (pd.DataFrame): input data frame to be cleaned and split.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.

    Raises:
        e: error in processing data cleaning and splitting.
    """
    try:
        # Process Data Cleaning
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        # Process Data Splitting
        divide_strategy = DataSplitStrategy()
        data_split = DataCleaning(processed_data, divide_strategy)
        X_train, X_val, y_train, y_val = data_split.handle_data()
        return X_train, X_val, y_train, y_val
    
        # Log Run
        logging.info("Data cleaning and splitting completed successfully.")
        
    except Exception as e:
        logging.error(f"Error in data cleaning step: {e}")
        raise e