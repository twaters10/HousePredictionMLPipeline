import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union
from sklearn.preprocessing import LabelEncoder

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
            # Impute missing values
            data["Area"].fillna(data["Area"].median())
            data["Bedrooms"].fillna(data["Bedrooms"].median())
            data["Bathrooms"].fillna(data["Bathrooms"].median())
            data["Floors"].fillna(data["Floors"].median())
            data["YearBuilt"].fillna(round(data["YearBuilt"].median()))
            
            # Create new features
            data['bedroom_bathroom_ratio'] = data['Bedrooms'] / data['Bathrooms']
            data['bedroom_floor_ratio'] = data['Bedrooms'] / data['Floors']
            
            # Label Encode categorical features
            data['Location_Label_Encoded'] = LabelEncoder().fit_transform(data['Location'])
            data['Condition_Label_Encoded'] = LabelEncoder().fit_transform(data['Condition'])
            data['Garage_Label_Encoded'] = LabelEncoder().fit_transform(data['Garage'])
            
            # Drop original categorical columns
            data.drop(['Location', 'Condition', 'Garage'], axis=1, inplace=True)
            return data
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise

class DataSplitStrategy(DataStrategy):
    """Concrete Strategy for splitting data into training and validation."""
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
        try:
            X = data.drop("Price", axis=1)
            y = data["Price"]
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