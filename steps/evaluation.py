import logging
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluate the trained machine learning model.
    
    This step is responsible for evaluating the model's performance
    and logging the results.
    
    Returns:
        None
    """
