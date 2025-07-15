import logging
import pandas as pd 
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train a machine learning model using the provided DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the training data.
    
    Returns:
        None
    """
