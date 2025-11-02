import mlflow
import mlflow.sklearn
import os
import shutil

# 🧹 Clean old MLflow directories (removes C: references)
if os.path.exists("mlruns"):
    shutil.rmtree("mlruns", ignore_errors=True)
os.makedirs("mlruns", exist_ok=True)

# 🧩 Ensure output dirs exist
os.makedirs("plots", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

import mlflow
import mlflow.sklearn
import platform
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("file:./mlruns")

import platform
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Clean up any old MLflow environment ---
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("MLFLOW_HOME", None)

# --- Set safe MLflow tracking path ---
mlflow.set_tracking_uri("file:./mlruns")

try:
    # Prepare data and model
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save metrics for DVC
    metrics = {"accuracy": float(accuracy)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    input_example = np.array([X_train[0]])

    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=clf,
            name="model",                 # updated API
            input_example=input_example
        )

        print(f"Model logged successfully with accuracy: {accuracy:.2f}")

    # ✅ Create plots folder for DVC output
    os.makedirs("plots", exist_ok=True)
    with open("plots/info.txt", "w") as f:
        f.write(f"Model accuracy: {accuracy:.2f}\n")

except Exception as e:
    print(f"Error in pipeline: {e}")
    import traceback
    traceback.print_exc()
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(clf, "artifacts/model.joblib")
    print("Model saved as artifacts/model.joblib (fallback)")

