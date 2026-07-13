import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Generate synthetic sensor readings
temperature = np.random.normal(loc=70, scale=10, size=n_samples)  # Celsius
vibration = np.random.normal(loc=50, scale=15, size=n_samples)    # Hz
pressure = np.random.normal(loc=100, scale=5, size=n_samples)     # PSI

# Introduce failure conditions (Target = 1) when temperature or vibration spikes
failure = (temperature > 85) | (vibration > 75)
failure = failure.astype(int)

# Create DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'failure': failure
})

# Save locally
df.to_csv('sensor_data.csv', index=False)
print("Dataset 'sensor_data.csv' created successfully!")