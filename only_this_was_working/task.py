"""test: A Flower / sklearn app."""

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression

from datasets import load_dataset

# Single file
data_files = "data/data.csv"
#dataset = load_dataset("csv", data_files=data_files)
fds = None

def load_data(partition_id: int, num_partitions: int):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    try:
        df = pd.read_csv(data_files)
    except FileNotFoundError:
        print(f"Error: File not found at {data_files}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


    # Simulate partitioning (IID)
    partition_size = len(df) // num_partitions
    start = partition_id * partition_size
    end = (partition_id + 1) * partition_size
    if partition_id == num_partitions - 1:  # Last partition takes the remainder
        end = len(df)

    partition = df.iloc[start:end].copy()  # Create a copy to avoid SettingWithCopyWarning


    X = partition.drop('diagnosis', axis=1).values
    y = partition['diagnosis'].values


    # Split the on edge data: 80% train, 20% test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test

def get_model(penalty: str, local_epochs: int):
    # Using best parameters from the provided info
    return LogisticRegression(
        penalty=penalty,
        C=1,  # Best parameter from results
        solver='lbfgs',  # Best parameter from results
        max_iter=local_epochs,
        warm_start=True,
    )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    n_classes = 2  # Binary classification (diagnosis)
    n_features = 30  # Number of features in dataset (all columns except diagnosis)
    model.classes_ = np.array([0, 1])  # Binary classification classes

    model.coef_ = np.zeros((1, n_features))  # Binary classification uses (1, n_features) shape
    if model.fit_intercept:
        model.intercept_ = np.zeros(1)  # Binary classification intercept
