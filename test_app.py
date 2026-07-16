from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import app as app_module

# 1. Initialize TestClient
client = TestClient(app_module.app)

# 2. Inject a mock model into app.py so tests never hit 500 "Model not loaded"
mock_model = MagicMock()
mock_model.predict.return_value = [0]
app_module.model = mock_model


def test_health_check():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running!"}


def test_prediction_endpoint():
    """Test the prediction endpoint with valid sensor payload data."""
    payload = {
        "temperature": 85.5,
        "vibration": 0.02,
        "pressure": 101.3,
        "rotational_speed": 1500.0
    }

    response = client.post("/predict", json=payload)

    # Assert successful response (200 OK)
    assert response.status_code == 200
    assert response.json() == {"prediction": 0}