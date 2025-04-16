# fl_client/data_utils.py
import numpy as np
import os
from dotenv import load_dotenv
from typing import Dict, Tuple

# Attempt to load dotenv, but don't fail if .env file doesn't exist
try:
    load_dotenv()
except Exception:
    pass

def get_client_data(instruction: dict, client_id: str):
    """
    Generates random data for client training.
    
    Parameters:
        instruction (dict): Instructions from server (can be empty)
        client_id (str): Unique identifier for this client
    
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    print(f"Client {client_id}: Generating random training data")
    
    # Convert client_id to an integer seed
    try:
        seed = int(client_id) % 10000
    except ValueError:
        # If client_id isn't numeric, hash it to get a numeric seed
        seed = hash(client_id) % 10000
    
    # Set seed for reproducibility but different per client
    np.random.seed(seed)
    
    # Generate random data: 100 samples with 5 features
    num_samples = 100
    num_features = 5
    X = np.random.rand(num_samples, num_features)
    
    # Create a linear relationship with noise to simulate regression problem
    # Each client gets slightly different coefficients to simulate heterogeneous data
    coefficients = np.array([1.0, -2.5, 3.0, -0.5, 1.5]) + (np.random.rand(num_features) * 0.5)
    intercept = 2.0 + (np.random.rand() - 0.5)
    
    # Generate target values with some noise
    y = X @ coefficients + intercept + np.random.randn(num_samples) * 0.5
    
    print(f"Client {client_id}: Generated {num_samples} training samples")
    return X, y

def get_client_data_from_partition(data_instruction: dict) -> Tuple[np.ndarray, np.ndarray]:
    # The server sends the partition as a dict with 'X' and 'y' as lists
    X = np.array(data_instruction['X'])
    y = np.array(data_instruction['y'])
    return X, y