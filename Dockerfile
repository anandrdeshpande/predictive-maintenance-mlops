# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our application code and MLflow runs
COPY app.py .
COPY mlruns/ ./mlruns/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to start the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]