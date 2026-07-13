import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import skops.io as sio

app = FastAPI(title="Predictive Maintenance API")

# Load our trained model artifact from MLflow (adjust run path as needed)
# For this example, we assume the model was saved locally or logged

# Path to your saved model file
model_path = (
    "mlruns/1/models/m-820413bb7e19466f8a9431d945683e7d/artifacts/model.skops"
)

# 1. Get the list of untrusted types found inside the model file
untrusted_types = sio.get_untrusted_types(file=model_path)

# 2. Pass those types as trusted to load the model securely
model = sio.load(model_path, trusted=untrusted_types)

#model = sio.load(
#    "mlruns/1/models/m-820413bb7e19466f8a9431d945683e7d/artifacts/model.skops",
#    trusted=True,
#)

#model = mlflow.pyfunc.load_model("mlruns/1/models/m-820413bb7e19466f8a9431d945683e7d/artifacts")
#model = mlflow.sklearn.load_model("mlruns/1/models/m-b13e5643d8414aff99d8135b8f91413e/artifacts")
#model = mlflow.sklearn.load_model("mlruns/1/YOUR_RUN_ID/artifacts/model")
# mlruns\1\models\m-b13e5643d8414aff99d8135b8f91413e\artifacts

# Define the input data schema
class SensorData(BaseModel):
    temperature: float
    vibration: float
    pressure: float


@app.get("/")
def home():
    return {"status": "API is running!"}


@app.post("/predict")
def predict(data: SensorData):
    # Convert input JSON to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Generate prediction (0 = Normal, 1 = Failure)
    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "status": "Failure Detected ⚠️" if prediction == 1 else "Normal Operational State ✅",
    }