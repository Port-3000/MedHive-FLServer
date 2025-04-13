# fl_client/client_logic.py
import flwr as fl
import numpy as np
from sklearn.linear_model import LinearRegression
from .linear_regression import get_params, set_params, train, test
from .data_utils import get_client_data
from typing import Dict, List, Tuple

class LinearRegressionClient(fl.client.NumPyClient):
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.model = LinearRegression()
        self.X_train = None
        self.y_train = None
        self.X_test = None # Optional: if client holds test data
        self.y_test = None # Optional: if client holds test data

    def load_data(self, instruction: Dict):
        """Load data based on server instruction."""
        print(f"Client {self.client_id} loading data...")
        X, y = get_client_data(instruction, self.client_id)
        if X is not None and y is not None:
            # Simple train/test split (adjust as needed)
            split_idx = int(len(X) * 0.8)
            self.X_train, self.X_test = X[:split_idx], X[split_idx:]
            self.y_train, self.y_test = y[:split_idx], y[split_idx:]
            print(f"Client {self.client_id}: Loaded {len(self.X_train)} train, {len(self.X_test)} test samples.")
        else:
             print(f"Client {self.client_id}: Failed to load data.")
             # Handle error state - cannot participate without data

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        print(f"Client {self.client_id}: Sending parameters")
        return get_params(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        print(f"Client {self.client_id}: Receiving parameters")
        set_params(self.model, parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        print(f"Client {self.client_id}: Starting fit (training)")
        # Load data if not already loaded (or reload based on config)
        if self.X_train is None:
            data_instruction = config.get("data_instruction", {})
            self.load_data(data_instruction)

        if self.X_train is None or self.y_train is None:
             print(f"Client {self.client_id}: No training data available. Skipping fit.")
             # Return current parameters and 0 samples, signalling failure/inability to train
             # Flower >1.7 requires ndarrays_to_parameters here:
             # return ndarrays_to_parameters(get_params(self.model)), 0, {"error": "No data"}
             # Older Flower versions might return ndarrays directly:
             return get_params(self.model), 0, {"error": "No data"}


        self.set_parameters(parameters) # Update model with global params
        self.model, loss = train(self.model, self.X_train, self.y_train) # Train locally
        print(f"Client {self.client_id}: Fit completed. Loss={loss:.4f}")

        new_params = get_params(self.model)
        num_examples = len(self.X_train)
        metrics = {"loss": float(loss)} # Convert loss to float for serialization

        # Return updated parameters, number of examples, and metrics
        # Flower >1.7 requires ndarrays_to_parameters here:
        # return ndarrays_to_parameters(new_params), num_examples, metrics
        # Older Flower versions might return ndarrays directly:
        return new_params, num_examples, metrics


    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        print(f"Client {self.client_id}: Starting evaluation")
        if self.X_test is None or self.y_test is None:
            print(f"Client {self.client_id}: No test data. Skipping evaluation.")
            return 0.0, 0, {"error": "No test data"}

        self.set_parameters(parameters) # Update model with global params
        loss, accuracy = test(self.model, self.X_test, self.y_test) # Evaluate
        print(f"Client {self.client_id}: Evaluate completed. Loss={loss:.4f}, Accuracy={accuracy:.4f}")

        num_examples = len(self.X_test)
        metrics = {"loss": float(loss), "accuracy": float(accuracy)} # Convert to float

        return float(loss), num_examples, metrics

# Function to start the client (called by BeeWare UI)
def start_fl_client(server_address: str, client_id: str):
    print(f"Starting Flower client {client_id} connecting to {server_address}")
    try:
        # Convert NumPyClient to Client using to_client() method
        client = LinearRegressionClient(client_id=client_id).to_client()
        
        # Use the recommended client startup approach
        fl.client.start_client(
            server_address=server_address,
            client=client,
        )
        print(f"Client {client_id} finished.")
    except Exception as e:
        print(f"Client {client_id} connection error: {e}")
        # Update UI here to show disconnection/error