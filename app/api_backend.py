"""
Minimal FastAPI backend for mobile sensor data, in-memory storage, activity computation, and simple stress detection.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# In-memory storage for latest sensor data
latest_sensor_data = {
    "acc_x": 0.0,
    "acc_y": 0.0,
    "acc_z": 0.0,
    "heart_rate": 0.0,
    "timestamp": 0
}

class SensorData(BaseModel):
    acc_x: float
    acc_y: float
    acc_z: float
    heart_rate: float
    timestamp: int

def compute_activity(acc_x: float, acc_y: float, acc_z: float) -> float:
    """Compute activity level from accelerometer data."""
    return float(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))

def detect_stress(heart_rate: float, activity: float) -> str:
    """Simple stress detection based on heart rate and activity."""
    # Example logic: high heart rate and low activity = stress
    if heart_rate > 100 and activity < 1.2:
        return "Stressed"
    else:
        return "Normal"

@app.post("/sensor-data")
async def receive_sensor_data(data: SensorData):
    global latest_sensor_data
    latest_sensor_data = data.dict()
    return {"status": "received"}

@app.get("/latest-sensor-data")
def get_latest_sensor_data():
    acc_x = latest_sensor_data["acc_x"]
    acc_y = latest_sensor_data["acc_y"]
    acc_z = latest_sensor_data["acc_z"]
    heart_rate = latest_sensor_data["heart_rate"]
    timestamp = latest_sensor_data["timestamp"]
    activity = compute_activity(acc_x, acc_y, acc_z)
    stress_state = detect_stress(heart_rate, activity)
    return {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "heart_rate": heart_rate,
        "timestamp": timestamp,
        "activity": activity,
        "stress_state": stress_state
    }
