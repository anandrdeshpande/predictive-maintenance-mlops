import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_NAME = "PredictiveMaintenanceModel"

def train_model(data_path="sensor_data.csv"):
    print(f"[INFO] Retraining model using dataset: {data_path}...")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=["failure"])
    y = df["failure"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Set Experiment
    mlflow.set_experiment("Predictive_Maintenance_Retrain")
    
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        # Log Params and Metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        
        # 3. Log and Register Model in MLflow Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME
        )
        print(f"[MLFLOW] Logged model version under '{MODEL_NAME}'")
        
        # 4. Assign '@champion' Alias to the Latest Registered Version
        client = MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME)[-1].version
        
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=latest_version
        )
        print(f"[REGISTRY] Updated alias '@champion' -> Version {latest_version}")
        print(f"[SUCCESS] Retraining complete! Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()