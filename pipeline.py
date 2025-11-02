import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import platform
import joblib
import json
import numpy as np

# --- MLflow tracking setup ---
if platform.system() == "Windows":
    mlflow.set_tracking_uri("file:./mlruns")  # relative safe path
else:
    mlflow.set_tracking_uri("file:./mlruns")  # also works on Linux (GitHub Actions)

# --- Start pipeline ---
try:
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save metrics for DVC
    metrics = {"accuracy": float(accuracy)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        # Add an example input for model signature inference
        input_example = np.array([X_train[0]])

        # Log model safely (no registered model name)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",  # relative, safe
            input_example=input_example
        )

        print(f"Model logged successfully with accuracy: {accuracy:.2f}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

except Exception as e:
    print(f"Error in pipeline: {e}")
    import traceback
    traceback.print_exc()

    # --- Fallback save ---
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(clf, "artifacts/model.joblib")
    print("Model saved as artifacts/model.joblib (fallback)")

