# Import necessary steps for the pipeline 
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train_old import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig


def training_pipeline(data_path: str):
    """
    Pipeline for training a machine learning model.
    
    This pipeline includes steps for ingesting data, cleaning data, training a model,
    and evaluating the model.
    """
    # Define the steps in the pipeline
    df = ingest_data(data_path=data_path)
    
    X_train, X_val, y_train, y_val = clean_data(df)
    
    config = ModelNameConfig()
    model = train_model(X_train = X_train, 
                        y_train = y_train, 
                        X_val = X_val, 
                        y_val = y_val, 
                        config = config)
    
    r2, rmse = evaluate_model(model, X_val, y_val)
    # The pipeline returns the evaluation metrics
    return r2, rmse
    
    

    