import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


class Evaluation(ABC):
    """ Abstract Class for defining model evaluation interface. """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate evaluation scores.

        Args:
            y_true (np.ndarray): True Labels
            y_pred (np.ndarray): Prediction Labels
        """
        pass
    
class MSE(Evaluation):
    """ Evaluation using Mean Squared Error. 
        Args:
            y_true (np.ndarray): True Labels
            y_pred (np.ndarray): Prediction Labels
        Returns:
            mse (float): Mean Squared Error
        Raises:
            Exception: If error in calculating MSE.    
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Mean Squared Error (MSE)...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e: 
            logging.error(f"Error in calculating MSE: {e}")
            raise e

class R2(Evaluation):
    """ Evaluation using R2 Score. 
    Args:
        y_true (np.ndarray): True Labels
        y_pred (np.ndarray): Prediction Labels
    Returns:
        r2 (float): R2 Score
    Raises:
        Exception: If error in calculating R2 Score.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score...")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e: 
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e
        
class RMSE(Evaluation):
    """ Evaluation using Root Mean Squared Error. 
    Args:
        y_true (np.ndarray): True Labels
        y_pred (np.ndarray): Prediction Labels
    Returns:
        rmse (float): Root Mean Squared Error
    Raises:
        Exception: If error in calculating RMSE.
    """  
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Root Mean Squared Error (RMSE)...")
            rmse = root_mean_squared_error(y_true,y_pred)
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e: 
            logging.error(f"Error in calculating RMSE: {e}")
            raise e