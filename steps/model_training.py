import logging
import pandas as pd 
from src.model_dev import RFRegressorModel
from typing_extensions import Annotated
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin
from .config import ModelNameConfig
import mlflow


def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                config: ModelNameConfig) -> RegressorMixin:
    """
    Train a machine learning model using the provided DataFrame.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
    
    Returns:
        None
    """
    try:
        model = None
        with mlflow.start_run():        
            if config.model_name == "RandomForestRegressor":
                model = RFRegressorModel()
                trained_model = model.train(X_train, y_train)
                return trained_model
            
            else:
                raise ValueError(f"Model {ModelNameConfig.model_name} is not supported.")
        
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e