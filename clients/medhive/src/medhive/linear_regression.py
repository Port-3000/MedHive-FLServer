# fl_client/linear_regression.py
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Tuple

def get_params(model: LinearRegression) -> List[np.ndarray]:
    """Gets model parameters as a list of NumPy ndarrays."""
    # Check if model has been fitted
    if not hasattr(model, 'coef_'):
        # Return empty arrays with correct shapes if model hasn't been trained
        n_features = 1  # Default, will be properly set when trained
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        return [np.zeros(n_features), np.array([0.0])] if model.fit_intercept else [np.zeros(n_features)]
        
    if model.fit_intercept:
        # Ensure coef_ is 2D for consistency if needed, though 1D is fine for LinearRegression
        # coef = model.coef_.reshape(1, -1) if model.coef_.ndim == 1 else model.coef_
        return [model.coef_, np.array([model.intercept_])] # Coefs and intercept
    else:
        return [model.coef_]

def set_params(model: LinearRegression, params: List[np.ndarray]) -> LinearRegression:
    """Sets model parameters from a list of NumPy ndarrays."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1][0] # Intercept stored as single-element array
    return model

def train(model: LinearRegression, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[LinearRegression, float]:
    """Train the linear regression model."""
    model.fit(X_train, y_train)
    # Calculate training loss (Mean Squared Error)
    y_pred = model.predict(X_train)
    loss = np.mean((y_pred - y_train)**2)
    return model, loss

def test(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
     """Evaluate the linear regression model."""
     y_pred = model.predict(X_test)
     loss = np.mean((y_pred - y_test)**2)
     # You could add other metrics like R^2 score if needed
     # accuracy = model.score(X_test, y_test) # R^2 score for LinearRegression
     accuracy = 0.0 # Placeholder if not using R^2
     return loss, accuracy