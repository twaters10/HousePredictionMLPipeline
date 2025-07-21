# import sys
# import os

# print(f"Current Working Directory (CWD): {os.getcwd()}")
# print(f"Script Path: {os.path.abspath(__file__)}")
# print("sys.path:")
# for p in sys.path:
#     print(f"  - {p}")
from zenml.client import Client
from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Run Pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="/Users/tawate/Documents/HousePredictionMLPipeline/data/olist_customers_dataset.csv")