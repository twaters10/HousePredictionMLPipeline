from zenml import pipelines
# Import necessary steps for the pipeline 
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipelines
def training_pipeline(data_path: str):
    """
    Pipeline for training a machine learning model.
    
    This pipeline includes steps for ingesting data, cleaning data, training a model,
    and evaluating the model.
    """
    # Define the steps in the pipeline
    df = ingest_data(data_path=data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)
    
    

    