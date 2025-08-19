import logging
import pandas as pd 
from typing_extensions import Annotated
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin
from .config import ModelNameConfig
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
import mlflow


def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                config: ModelNameConfig,
                hyperparameters = None,) -> RegressorMixin:
    """
    Train a machine learning model using the provided DataFrame.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        config (ModelNameConfig): Configuration for the model.
        hyperparameters (dict, optional): Hyperparameters for the model. Defaults to None.
    Returns:
        None
    """
    try:
        model = None
        with mlflow.start_run():        
            if config.model_name == "RandomForestRegressor":
                # If hyperparameters are provided, use them
                if hyperparameters:
                    model = RandomForestRegressor(**hyperparameters)
                else:
                    model = RandomForestRegressor()
                # Train the model
                logging.info(f"Training model: {config.model_name} with hyperparameters: {hyperparameters}")
                # Fit the model
                if hyperparameters:
                    trained_model = model.fit(X_train, y_train, **hyperparameters)
                else:
                    trained_model = model.fit(X_train, y_train)
                logging.info("Model trained successfully.")
                return trained_model
            else:
                raise ValueError(f"Model {ModelNameConfig.model_name} is not supported.")
        
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
    
def tune_model(X_train: pd.DataFrame,
               y_train: pd.Series,
               X_val: pd.DataFrame,
               y_val: pd.Series,
               config: ModelNameConfig,
               search_space: dict = None) -> dict:
    """ Tune a machine learning model using the provided DataFrame.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
    Returns:
        dict: Best hyperparameters found during tuning.
    """
    # Define objective function
    def objective_rf(params):
        with mlflow.start_run(nested=True):
            reg = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=42
            )
            
            reg.fit(X_train, y_train)
            r2_score = reg.score(X_val, y_val)

            return {'loss': -r2_score, 'status': STATUS_OK}
    
    
    try:
        with mlflow.start_run():        
            if config.model_name == "RandomForestRegressor":
                # Define the search space for hyperparameters
                if search_space:
                    space = search_space
                else:
                    space = {
                        'n_estimators': hp.choice('n_estimators', range(10, 300)),
                        'max_depth': hp.choice('max_depth', range(1, 20)),
                        'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
                        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10))
                    }
        # Run optimization
        if config.model_name == "RandomForestRegressor":
            trials = Trials()
            logging.info("Starting hyperparameter tuning for RandomForestRegressor.")
            with mlflow.start_run():
                best_params = fmin(
                    fn=objective_rf,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50,
                    trials=trials
            )
            print("Best parameters found:", best_params)
    except Exception as e:
        logging.error(f"Error in tuning model: {e}")
        raise e
    return best_params
                
