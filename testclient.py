import requests

# The endpoint URL
url = "http://127.0.0.1:8000/predict"

# Your sensor payload
data = {"temperature": 92.5, "vibration": 80.1, "pressure": 102.3}

# Send the POST request
response = requests.post(url, json=data)

# Print the model's prediction
print(response.json())