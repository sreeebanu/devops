import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.datasets import make_classification

# Set MLflow tracking URI to local directory
mlflow.set_tracking_uri("./mlruns")
os.makedirs("./mlruns", exist_ok=True)

# Generate synthetic dataset with enough samples
data_X, data_y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

X = pd.DataFrame(data_X, columns=['feature1', 'feature2'])
y = pd.Series(data_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    n_estimators = 20
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Predictions:", preds)
    print("Actual:", y_test.values)
    print(f"Accuracy: {acc}")

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_metric('accuracy', acc)
    mlflow.sklearn.log_model(clf, 'model')

    os.makedirs('model', exist_ok=True)
    joblib.dump(clf, 'model/model.pkl')

    print(f"Logged model with accuracy: {acc}")

