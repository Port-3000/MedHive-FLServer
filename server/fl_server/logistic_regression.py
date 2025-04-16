import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from typing import List, Tuple

def get_params(model: LogisticRegression) -> List[np.ndarray]:
    if not hasattr(model, 'coef_'):
        n_features = 1
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        return [np.zeros((1, n_features)), np.zeros(1)]
    return [model.coef_, model.intercept_]

def set_params(model: LogisticRegression, params: List[np.ndarray]) -> LogisticRegression:
    model.coef_ = params[0]
    model.intercept_ = params[1]
    return model

def train(model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[LogisticRegression, float]:
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_train)
    loss = log_loss(y_train, y_pred_proba)
    return model, loss

def test(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    return loss, accuracy, precision, recall, f1
