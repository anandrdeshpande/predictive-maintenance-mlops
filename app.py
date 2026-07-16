import os
import mlflow.pyfunc
import skops.io as sio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Predictive Maintenance API")

# URI pointing to the model tagged with alias '@champion'
MODEL_URI = "models:/PredictiveMaintenanceModel@champion"
FALLBACK_MODEL_PATH = "model.skops"

model = None

# Load the model directly from MLflow Registry
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"[LOADED] Successfully loaded active model from: {MODEL_URI}")
except Exception as e:
    model = None
    print(f"[INFO] MLflow Registry model not found ({e}). Trying fallback root model...")

# 2. Fallback to root model.skops for local tests & CI/CD
    if os.path.exists(FALLBACK_MODEL_PATH):
        try:
            model = sio.load(FALLBACK_MODEL_PATH, trusted=True)
            print(f"[LOADED] Fallback model loaded from: {FALLBACK_MODEL_PATH}")
        except Exception as err:
            print(f"[ERROR] Could not load fallback model: {err}")

# Define input schema matching your dataset features
class SensorData(BaseModel):
    temperature: float
    vibration: float
    pressure: float



@app.get("/")
def home():
    return {"status": "API is running!"}

#@app.get("/")
#def read_root():
#    return {
#        "status": "API is running!",
#        "active_model_uri": MODEL_URI
#    }

@app.post("/predict")
def predict(data: SensorData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert input pydantic payload into a pandas DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict using the loaded MLflow PyFunc model
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}