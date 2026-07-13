import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 1. Load data
# x is the feature matrix
# y is the target variable

df = pd.read_csv("sensor_data.csv")
X = df[["temperature", "vibration", "pressure"]]
y = df["failure"]

# 2. Chronological / Simple Split
# X_train is the feature matrix
# X_test is the target variable
# y_train is the feature matrix
# y_test is the target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Set MLflow Experiment
mlflow.set_experiment("predictive_maintenance_v1")

with mlflow.start_run():
    # Model parameters
    n_estimators = 200
    max_depth = 10

    # Train model
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameters and metrics to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save model artifact
    mlflow.sklearn.log_model(clf, "model")

    print(f"Run complete! F1 Score: {f1:.2f}")