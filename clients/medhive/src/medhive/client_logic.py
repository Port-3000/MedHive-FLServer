# fl_client/client_logic.py
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from .logistic_regression import get_params, set_params, train, test
from .data_utils import get_client_data_from_partition
from typing import Dict, List, Tuple

class LogisticRegressionClient(fl.client.NumPyClient):
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self.X_train = None
        self.y_train = None

    def load_data(self, instruction: Dict):
        print(f"Client {self.client_id} loading data partition from server...")
        X, y = get_client_data_from_partition(instruction)
        self.X_train, self.y_train = X, y
        print(f"Client {self.client_id}: Loaded {len(self.X_train)} train samples.")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        print(f"Client {self.client_id}: Sending parameters")
        return get_params(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        print(f"Client {self.client_id}: Receiving parameters")
        set_params(self.model, parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        print(f"Client {self.client_id}: Starting fit (training)")
        if self.X_train is None:
            data_instruction = config.get("data_instruction", {})
            self.load_data(data_instruction)
        if self.X_train is None or self.y_train is None:
            print(f"Client {self.client_id}: No training data available. Skipping fit.")
            return get_params(self.model), 0, {"error": "No data"}
        self.set_parameters(parameters)
        self.model, loss = train(self.model, self.X_train, self.y_train)
        print(f"Client {self.client_id}: Fit completed. Loss={loss:.4f}")
        new_params = get_params(self.model)
        num_examples = len(self.X_train)
        metrics = {"loss": float(loss)}
        return new_params, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        print("parameters", parameters)
        print(f"Client {self.client_id}: No test data. Skipping evaluation.")
        return 0.0, 0, {"error": "No test data (server evaluates)"}


def start_fl_client(server_address: str, client_id: str):
    print(f"Starting Flower client {client_id} connecting to {server_address}")
    try:
        client = LogisticRegressionClient(client_id=client_id).to_client()
        fl.client.start_client(
            server_address=server_address,
            client=client,
        )
        print(f"Client {client_id} finished.")
    except Exception as e:
        print(f"Client {client_id} connection error: {e}")