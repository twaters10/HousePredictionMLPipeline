import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

""" Notes:
- This module defines strategies for data preprocessing and splitting.
- The `DataStrategy` abstract class defines the interface for data handling strategies.
- The `DataPreProcessStrategy` and `DataSplitStrategy` concrete strategies handle data preprocessing and splitting, respectively.
- The `DataCleaning` class orchestrates the data cleaning and splitting process.
- This is a strategy design pattern implementation for handling data in a flexible and reusable manner.
"""
class DataStrategy(ABC):
    """Abstract Class for defining strategy for handling data.

    Args:
        ABC (_type_): _description_
    """
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessStrategy(DataStrategy):
    """Concrete Strategy for preprocessing data."""
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Preprocess Data """
        try:
            # Drop unnecessary columns
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1)
            
            # Impute missing values
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)
            
            # Select only numeric columns
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(columns=cols_to_drop, axis=1)
            return data
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise

class DataSplitStrategy(DataStrategy):
    """Concrete Strategy for splitting data into training and validation."""
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_val, y_train, y_val
        
        except Exception as e:
            logging.error(f"Error in data splitting: {e}")
            raise
            
class DataCleaning:
    """
    Orchestrates data cleaning and divide into training and validation sets.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the chosen strategy"""
        return self.strategy.handle_data(self.data)