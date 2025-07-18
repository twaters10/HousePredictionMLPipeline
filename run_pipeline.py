import sys
import os

print(f"Current Working Directory (CWD): {os.getcwd()}")
print(f"Script Path: {os.path.abspath(__file__)}")
print("sys.path:")
for p in sys.path:
    print(f"  - {p}")

from pipeline.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Run Pipeline
    training_pipeline(data_path="olist_customers_dataset.csv")