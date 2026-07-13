import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import skops.io as sio

def train_model(data_path="sensor_data.csv", model_output_path="model.skops"):
    # Plain text print statements without unicode emojis
    print(f"[INFO] Retraining model using dataset: {data_path}...")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # Target column
    X = df.drop(columns=["failure"])
    y = df["failure"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Start MLflow Run
    mlflow.set_experiment("Predictive_Maintenance_Retrain")
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        # Log metrics to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"[SUCCESS] Retraining complete! Test Accuracy: {accuracy:.4f}")
        
        # 3. Save model using skops
        sio.dump(model, model_output_path)
        print(f"[SAVED] Saved newly trained model to '{model_output_path}'")
        
        # 2. ALSO log the model artifact inside mlruns for historical tracking
        mlflow.log_artifact(model_output_path)
        print(f"[MLFLOW] Archived model artifact in mlruns/")

if __name__ == "__main__":
    train_model()