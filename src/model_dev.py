import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """ Abstract Class for defining a model interface. """
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train model

        Args:
            X_train (pd.DataFrame): Training Data
            y_train (pd.Series): Label Data
        """
        pass
    
class LinearRegressionModel(Model):
    """ Linear Regression Model Implementation. """
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Train the linear regression model.

        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
            
        Returns:
            None

        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Linear Regression model trained successfully.")        
            return reg
        
        except Exception as e:
            logging.error(f"Error in training Linear Regression model: {e}")
            raise e
        