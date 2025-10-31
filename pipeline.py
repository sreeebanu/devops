import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import platform

# Set MLflow tracking URI based on the operating system
if platform.system() == "Windows":
    mlflow.set_tracking_uri("mlruns")  # Relative path for Windows
else:
    mlflow.set_tracking_uri("./mlruns")  # Relative path for Linux (GitHub Actions)

# Alternative: Use absolute path that works on both systems
# import pathlib
# mlflow_dir = pathlib.Path("mlruns").absolute()
# mlflow.set_tracking_uri(mlflow_dir.as_uri())

try:
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model with proper settings
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name="sklearn-random-forest"
        )
        
        print(f"Model logged successfully with accuracy: {accuracy:.2f}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
except Exception as e:
    print(f"Error in pipeline: {e}")
    import traceback
    traceback.print_exc()
    # Fallback: save model without MLflow
    import joblib
    joblib.dump(clf, 'model.joblib')
    print("Model saved as model.joblib (fallback)")
