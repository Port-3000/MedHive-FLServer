# fl_server/fl_logic.py
import flwr as fl
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import mlflow
import os
from dotenv import load_dotenv
from .database import (
    get_supabase_client,
    update_round_status,
    update_aggregated_model_path,
    log_model_performance,
    create_fl_round_in_db,
    get_active_model_info
)
import pickle # Or joblib
from .data_utils import load_and_preprocess_data, partition_data
from .logistic_regression import get_params, set_params, test as test_model
from sklearn.linear_model import LogisticRegression

load_dotenv()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000") # Default to local
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Configuration ---
MODEL_NAME = "linear-regression-example"
MODEL_VERSION = "1.0"
MLFLOW_EXPERIMENT_NAME = "Federated Linear Regression"
MIN_CLIENTS_PER_ROUND = 2 # Example: require at least 2 clients

# Ensure experiment exists
try:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
except Exception as e:
    print(f"Error setting up MLflow experiment: {e}")
    experiment_id = None

current_fl_round_id = 10
current_model_id = 69

# --- Data Loading and Partitioning ---
# Path to your data file (update as needed)
DATA_PATH = os.environ.get("SERVER_DATA_PATH", "data/data.csv")
NUM_CLIENTS = MIN_CLIENTS_PER_ROUND
X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
train_partitions = partition_data(X_train, y_train, NUM_CLIENTS)

# --- Initialize server-side model with correct shape ---
# Fit on a single batch to set coef_ and intercept_ shapes
if X_train.shape[0] > 0:
    server_init_model = LogisticRegression()
    # Use at least one sample from each class if possible
    unique_classes = np.unique(y_train)
    if len(unique_classes) > 1:
        # Take one sample from each class for fitting
        idxs = []
        for cls in unique_classes:
            idxs.append(np.where(y_train == cls)[0][0])
        X_init = X_train[idxs]
        y_init = y_train[idxs]
    else:
        X_init = X_train[:1]
        y_init = y_train[:1]
    server_init_model.fit(X_init, y_init)
else:
    server_init_model = LogisticRegression()

# Custom Strategy to handle MLflow logging and Supabase updates
class FedLinearRegressionStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_server_round = 0
        self.active_model_info = get_active_model_info(MODEL_NAME, MODEL_VERSION)
        if self.active_model_info:
            global current_model_id
            current_model_id = self.active_model_info['id']
        else:
            print(f"CRITICAL: No active model found for {MODEL_NAME} v{MODEL_VERSION} in Supabase.")
            current_model_id = None
        # Store partition info for each round
        self.train_partitions = train_partitions

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training."""
        self.current_server_round = server_round
        global current_fl_round_id
        global current_model_id

        if not current_model_id:
            print("Skipping round: No active model ID.")
            return []

        # Create FL Round entry in Supabase for this new round
        current_fl_round_id = create_fl_round_in_db(current_model_id, server_round)
        if not current_fl_round_id:
            print(f"Failed to create database entry for round {server_round}. Skipping round.")
            return []

        update_round_status(current_fl_round_id, "in_progress")
        print(f"--- Starting FL Round {server_round} (DB ID: {current_fl_round_id}) ---")

        # Start MLflow run for the round
        with mlflow.start_run(run_name=f"FL_Round_{server_round}", nested=True) as run:
            mlflow.log_param("round_number", server_round)
            mlflow.log_param("fl_round_db_id", current_fl_round_id)
            mlflow.log_param("model_db_id", current_model_id)
            mlflow.log_param("strategy", "FedAvg") # Or self.__class__.__name__
            mlflow.set_tag("mlflow.runName", f"FL_Round_{server_round}") # Ensure name sticks

        # Assign each client a partition index
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_available_clients
        )
        fit_ins_list = []
        for idx, client in enumerate(clients):
            config = {"partition_index": idx}
            fit_ins = fl.common.FitIns(parameters, config)
            fit_ins_list.append((client, fit_ins))
        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            # Potentially update round status to failed in Supabase
            if current_fl_round_id:
                update_round_status(current_fl_round_id, "failed")
            return None, {}

        # Log client metrics to MLflow (collected from FitRes)
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id, nested=True): # Re-open parent round run
            total_num_examples = sum([fit_res.num_examples for _, fit_res in results])
            for _, fit_res in results:
                if fit_res.metrics:
                    # Log individual client metrics if desired (can get verbose)
                    # mlflow.log_metrics({f"client_{fit_res.cid}_loss": fit_res.metrics.get("loss", 0)}, step=server_round)
                    pass # Decide if needed

            # Aggregate metrics (simple average here, could be weighted)
            aggregated_loss = np.mean([fit_res.metrics.get("loss", 0.0) for _, fit_res in results if fit_res.metrics])
            mlflow.log_metric("aggregated_fit_loss", aggregated_loss, step=server_round)
            print(f"Round {server_round} aggregated fit loss: {aggregated_loss}")

        # TODO: Update fl_participants status to 'completed' in Supabase for successful clients

        # Call FedAvg's aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # You might add custom aggregated metrics here if needed

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            # Potentially update round status to failed in Supabase if eval was mandatory
            return None, {}

        # Aggregate loss using weighted average logic from Flower base strategy
        loss_aggregated = fl.server.strategy.aggregate.weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )
        # Aggregate custom metrics (e.g., accuracy - needs clients to return it)
        metrics_aggregated = {}
        # Example for accuracy:
        # accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results if "accuracy" in r.metrics]
        # examples = [r.num_examples for _, r in results if "accuracy" in r.metrics]
        # if examples:
        #     metrics_aggregated["accuracy"] = sum(accuracies) / sum(examples)

        print(f"Round {server_round} aggregated evaluation loss: {loss_aggregated}")
        # if metrics_aggregated:
        #      print(f"Round {server_round} aggregated evaluation metrics: {metrics_aggregated}")

        # Log aggregated evaluation metrics to MLflow
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id, nested=True): # Re-open parent round run
            mlflow.log_metric("aggregated_eval_loss", loss_aggregated, step=server_round)
            # if metrics_aggregated:
            #     mlflow.log_metrics(metrics_aggregated, step=server_round)

        # Log to Supabase model_performance table
        if current_model_id and current_fl_round_id:
            # Combine loss and other metrics
            db_metrics = {"loss": loss_aggregated}
            # db_metrics.update(metrics_aggregated) # Add accuracy etc. if calculated
            log_model_performance(current_model_id, current_fl_round_id, db_metrics, test_dataset_id=None) # Add test dataset ID if applicable

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f"--- Evaluating global model after round {server_round} ---")
        try:
            # Convert parameters to numpy arrays and set to a LogisticRegression model
            param_ndarrays = parameters_to_ndarrays(parameters)
            model = LogisticRegression()
            model = set_params(model, param_ndarrays)
            loss, accuracy, precision, recall, f1 = test_model(model, X_test, y_test)
            with mlflow.start_run(run_id=mlflow.active_run().info.run_id, nested=True):
                mlflow.log_metric("server_eval_loss", loss, step=server_round)
                mlflow.log_metric("server_eval_accuracy", accuracy, step=server_round)
                mlflow.log_metric("server_eval_precision", precision, step=server_round)
                mlflow.log_metric("server_eval_recall", recall, step=server_round)
                mlflow.log_metric("server_eval_f1", f1, step=server_round)
            if current_model_id and current_fl_round_id:
                log_model_performance(current_model_id, current_fl_round_id, {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}, test_dataset_id=None)
            return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        except Exception as e:
            print(f"Server-side evaluation failed: {e}")
            return None

# --- Helper Functions for Parameter Conversion (for Scikit-learn) ---
def ndarrays_to_parameters(ndarrays: List[np.ndarray]) -> Parameters:
    """Convert NumPy ndarrays to Flower Parameters object."""
    tensors = [fl.common.ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

def parameters_to_ndarrays(parameters: Parameters) -> List[np.ndarray]:
    """Convert Flower Parameters object to NumPy ndarrays."""
    return [fl.common.bytes_to_ndarray(tensor) for tensor in parameters.tensors]

# --- MLflow Model Saving ---
def save_model_mlflow(model_params: List[np.ndarray], server_round: int):
    if not experiment_id or not current_fl_round_id:
        print("Cannot save model: Missing experiment ID or FL round ID.")
        return None

    # Save parameters (e.g., using pickle or joblib)
    model_dir = "fl_models_temp"
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"linear_regression_params_round_{server_round}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(model_params, f)

    # Log as artifact in the current round's MLflow run
    try:
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id, nested=True): # Re-open parent round run
            mlflow.log_artifact(model_path, artifact_path="model_parameters")
            print(f"Model parameters for round {server_round} saved to MLflow artifact.")

            # Construct the artifact URI (may depend on MLflow backend)
            artifact_uri = f"runs:/{mlflow.active_run().info.run_id}/model_parameters/{model_filename}"

            # Update Supabase fl_rounds table with the path/URI
            update_aggregated_model_path(current_fl_round_id, artifact_uri)

            # Clean up local temp file
            os.remove(model_path)
            return artifact_uri

    except Exception as e:
        print(f"Error saving model artifact to MLflow: {e}")
        return None

# --- FL Server Main Setup ---
def start_fl_server(num_rounds: int = 3):
    global current_fl_round_id # Ensure global is accessible
    global current_model_id

    if not current_model_id:
        print("Cannot start FL Server: No active model configured in Supabase.")
        return

    # Define strategy
    strategy = FedLinearRegressionStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for fitting
        min_fit_clients=MIN_CLIENTS_PER_ROUND,
        min_available_clients=MIN_CLIENTS_PER_ROUND, # Wait for minimum clients
        # fraction_evaluate=0.5, # Example: Evaluate on 50% of clients
        # min_evaluate_clients=1, # Example: Minimum 1 client for evaluation
        # evaluate_fn=None, # Using client-side or server-side eval within strategy
        # on_fit_config_fn=fit_config, # Defined in strategy
        # on_evaluate_config_fn=evaluate_config, # Pass eval config if needed
        # initial_parameters=None # Load initial params if needed
    )

    print(f"Starting MLflow run for the overall FL session (Experiment: {MLFLOW_EXPERIMENT_NAME})...")
    with mlflow.start_run(run_name="Federated_Linear_Regression_Run") as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_param("num_rounds", num_rounds)
        mlflow.log_param("min_clients_per_round", MIN_CLIENTS_PER_ROUND)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_version", MODEL_VERSION)
        mlflow.log_param("supabase_model_id", current_model_id)
        print(f"Parent MLflow Run ID: {parent_run_id}")

        # Start Flower server
        history = fl.server.start_server(
            server_address="0.0.0.0:8089",  # Changed port to 8081
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        print("FL Server finished.")
        print("History:", history) # History object contains aggregated metrics over rounds

        # Log final aggregated metrics from history if needed
        if history.losses_distributed:
            mlflow.log_metric("final_avg_loss_distributed", np.mean(history.losses_distributed))
        if history.metrics_distributed.get("accuracy"): # If accuracy was aggregated
            mlflow.log_metric("final_avg_accuracy_distributed", np.mean(history.metrics_distributed["accuracy"]))

        # --- Final Model Saving ---
        # Get the final aggregated parameters from the strategy if available
        final_params = strategy.aggregate_fit(num_rounds, [], [])[0] # A bit hacky, might need cleaner way
        if final_params:
            print("Saving final aggregated model parameters to MLflow...")
            final_params_nd = parameters_to_ndarrays(final_params)
            final_model_path = save_model_mlflow(final_params_nd, server_round=num_rounds)
            if final_model_path:
                print(f"Final model parameters saved: {final_model_path}")
                # Update the main ml_models table in Supabase with the final path/performance
                final_metrics = {"loss": history.losses_distributed[-1][1] if history.losses_distributed else None} # Add other final metrics
                supabase_client = get_supabase_client()
                try:
                    supabase_client.table("ml_models").update({
                        "model_path": final_model_path,
                        "status": "active", # Or 'completed_training'
                        "accuracy": final_metrics.get("accuracy"), # If calculated
                        "loss": final_metrics.get("loss"),
                        # Add other scores if available
                        "updated_at": "now()"
                    }).eq("id", current_model_id).execute()
                    print("Updated main model entry in Supabase.")
                except Exception as e:
                    print(f"Error updating main model entry in Supabase: {e}")
            else:
                print("Failed to save final model.")
        else:
            print("No final parameters available to save.")
