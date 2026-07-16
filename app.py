import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Predictive Maintenance API")

# URI pointing to the model tagged with alias '@champion'
MODEL_URI = "models:/PredictiveMaintenanceModel@champion"

# Load the model directly from MLflow Registry
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"[LOADED] Successfully loaded active model from: {MODEL_URI}")
except Exception as e:
    model = None
    print(f"[WARNING] Could not load model from registry: {e}")

# Define input schema matching your dataset features
class SensorData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

@app.get("/")
def read_root():
    return {
        "status": "API is running!",
        "active_model_uri": MODEL_URI
    }

@app.post("/predict")
def predict(data: SensorData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert input pydantic payload into a pandas DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict using the loaded MLflow PyFunc model
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}