import pytest
from fastapi.testclient import TestClient
from app import app  # Imports your FastAPI instance

# Initialize the TestClient
client = TestClient(app)

def test_health_check():
    """Test that the root/health endpoint returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running!"}  # Added '!'
    
def test_prediction_endpoint():
    """Test the prediction endpoint with valid sensor payload data."""
    payload = {
        "temperature": 85.5,
        "vibration": 0.02,
        "pressure": 101.3,
        "rotational_speed": 1500
    }
    
    response = client.post("/predict", json=payload)
    
    # Assert successful response
    assert response.status_code == 200
    
    # Assert expected keys in response JSON
    data = response.json()
    assert "prediction" in data