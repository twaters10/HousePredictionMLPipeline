import logging
import pandas as pd
from zenml import step
from src.evaluate_scores import Evaluation, MSE, RMSE, R2
from sklearn.base import BaseEstimator, RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
import mlflow
from zenml.client import Client

expierment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=expierment_tracker.name)
def evaluate_model(model: RegressorMixin, 
                   X_val: pd.DataFrame, 
                   y_val: pd.Series) -> Tuple[Annotated[float, "r2_score"],
                                              Annotated[float, "rmse_score"]
                                        ]:
    """
    Evaluate the trained machine learning model.
    
    This step is responsible for evaluating the model's performance
    and logging the results.
    
    Args:
        model (RegressorMixin): The trained machine learning model.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
    
    Returns:
        Tuple[float, float]: A tuple containing the R2 score and RMSE.
    """
 
    try:
        # Calculate Precitions
        predictions = model.predict(X_val)
        
        # Initialize evaluation metrics
        mse = MSE()
        rmse = RMSE()
        r2 = R2()

        # Evaluate the model
        mse_score = mse.calculate_scores(y_val, predictions)
        rmse_score = rmse.calculate_scores(y_val, predictions)
        r2_score = r2.calculate_scores(y_val, predictions)

        # Log the evaluation results
        logging.info(f"MSE: {mse_score}")
        logging.info(f"RMSE: {rmse_score}")
        logging.info(f"R2: {r2_score}")
        mlflow.log_metric("mse", mse_score)
        mlflow.log_metric("rmse", rmse_score)
        mlflow.log_metric("r2", r2_score)
        
        return r2_score, rmse_score

    except Exception as e:
        logging.error(f"Error in model evaluation step: {e}")
        raise e
