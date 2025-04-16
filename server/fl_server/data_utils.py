import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List

# Load and preprocess the data (as in test.py)
def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    X = X.fillna(X.mean())
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train.values, y_test.values

# Partition the training data for each client
def partition_data(X_train: np.ndarray, y_train: np.ndarray, num_clients: int) -> List[Dict[str, np.ndarray]]:
    partitions = []
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    for idx in split_indices:
        partitions.append({
            'X': X_train[idx],
            'y': y_train[idx]
        })
    return partitions
