import mlflow.sklearn
from steps.s3_ingest_data import *
from steps.clean_data import *
from config.access_keys import *
import pandas as pd
import logging
from steps.model_training import train_model, tune_model
from steps.config import ModelNameConfig
from hyperopt import hp
import mlflow

if __name__ == "__main__":
    logging.info("Starting S3CSVReader...")
    reader = S3CSVReader(bucket_name=S3_BUCKET_NAME, region_name=AWS_REGION, 
                         aws_access_key_id=S3_AWS_ACCESS_KEY_ID, aws_secret_access_key=S3_AWS_SECRET_ACCESS_KEY)
    # Read the CSV file from S3
    df = reader.read_csv(s3_key=S3_KEY, encoding='utf-8')
    
    # Clean, transform, and split the data.
    processed_df = clean_data(df)
    X_train, X_val, y_train, y_val = split_data(processed_df)
    
    # Load the processed data back to S3
    load_processed_data_to_s3(processed_df, bucket_name=S3_BUCKET_NAME, csvfilename='processed_house_prices.csv')
    
    
    # Train Model
    config = ModelNameConfig()
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name="rf_regressor_experiment_8192025")
    mlflow.sklearn.autolog(silent=True)
    model = train_model(X_train = X_train, 
                        y_train = y_train, 
                        config = config)
    
    # Log the sklearn model and register it in MLflow
    logging.info("Logging the model to MLflow...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="rf_regressor_v1",
        input_example=X_train,
        registered_model_name="Best_RF_House_Model"
    )
    logging.info("Model logged successfully.")
    
    
    # Tune Model
    # config = ModelNameConfig()
    # search_space =  {
    #                 'n_estimators': hp.choice('n_estimators', range(100, 500)),
    #                 'max_depth': hp.choice('max_depth', range(1, 20)),
    #                 'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
    #                 'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10))
    #                 }
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # mlflow.set_experiment(experiment_name="rf_regressor_experiment_8192025")
    # mlflow.sklearn.autolog(silent=True)
    # params = tune_model(X_train = X_train, 
    #                     y_train = y_train,
    #                     X_val = X_val,
    #                     y_val = y_val, 
    #                     config = config,
    #                     search_space=search_space)
    
    # Train Model on best params and save to pkl
    # config = ModelNameConfig()
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # mlflow.set_experiment(experiment_name="rf_regressor_experiment_8192025")
    # mlflow.sklearn.autolog()
    # tuned_model = train_model(
    #                 X_train,
    #                 y_train,
    #                 hyperparameters=params,
    #                 config=config,
    #             )

    
    