import logging
import pandas as pd 
from zenml import step
from src.model_dev import LinearRegressionModel
from typing_extensions import Annotated
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

expierment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=expierment_tracker.name)
def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                y_val: pd.Series,
                X_val: pd.DataFrame,
                config: ModelNameConfig) -> RegressorMixin:
    """
    Train a machine learning model using the provided DataFrame.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
    
    Returns:
        None
    """
    try:
        model = None
        config = ModelNameConfig()
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        
        else:
            raise ValueError(f"Model {ModelNameConfig.model_name} is not supported.")
        
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e