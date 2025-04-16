import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import pickle
import os

# Load data
data_path = 'data/data.csv'
df = pd.read_csv(data_path)

# Preprocess
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
X = X.fillna(X.mean())
y = df['diagnosis']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline with logistic regression parameters
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    ))
])

# Train
print("Training logistic regression model...")
pipeline.fit(X_train, y_train)

# Test and evaluate
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
loss = log_loss(y_test, y_proba)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Log Loss: {loss:.4f}")

# Save the model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "logistic_regression_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"\nModel saved to {model_path}")

print("\nThis is the centralized training benchmark. For federated learning:")
print("1. Start the server: python -m server.app")
print("2. Connect multiple clients using the client application")
