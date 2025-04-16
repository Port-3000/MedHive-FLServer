"""test: A Flower / sklearn app."""

from typing import Dict, List, Optional, Tuple, Union

from flwr.common import Context, Metrics, FitRes, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from only_this_was_working.task import get_model, get_model_params, set_initial_params


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy that prints accuracy after each round."""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, FitRes]],
        failures: List[Union[Tuple[int, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model weights and print round information."""
        # Call the parent's aggregate_fit method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Print round information
        print(f"\n--- Round {server_round} completed ---")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> tuple:
        """Aggregate evaluation metrics and print them."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        # Fix: Use (ClientProxy, EvaluateRes) tuple structure
        accuracies = [
            r[1].metrics["accuracy"] * r[1].num_examples
            for r in results
            if r[1].metrics and "accuracy" in r[1].metrics
        ]
        examples = [
            r[1].num_examples
            for r in results
            if r[1].metrics and "accuracy" in r[1].metrics
        ]
        if examples:
            avg_accuracy = sum(accuracies) / sum(examples)
            print(f"Round {server_round} - Average accuracy: {avg_accuracy:.4f}")
            aggregated_metrics["accuracy"] = avg_accuracy
        return aggregated_loss, aggregated_metrics


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy with our custom FedAvg implementation
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
