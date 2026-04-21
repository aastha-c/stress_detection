import requests
import random
import time
from datetime import datetime

API_URL = "http://127.0.0.1:8000/sensor-data"


def generate_sensor_data():
    data = {
        "heart_rate": random.randint(65, 110),
        "temperature": round(random.uniform(36.2, 37.5), 2),
        "acc_x": round(random.uniform(-1, 1), 3),
        "acc_y": round(random.uniform(-1, 1), 3),
        "acc_z": round(random.uniform(-1, 1), 3),
        "timestamp": int(time.time())
    }
    return data


def main():
    print("Starting wearable sensor data simulation...")
    while True:
        sensor_data = generate_sensor_data()
        print(f"Sending data: {sensor_data}")
        try:
            response = requests.post(API_URL, json=sensor_data)
            print(f"Response: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Error sending data: {e}")
        time.sleep(2)


if __name__ == "__main__":
    main()
