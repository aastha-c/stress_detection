from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from your HTML page

@app.route('/sensor-data', methods=['POST'])
def sensor_data():
    data = request.get_json()
    # Example: calculate a dummy stress level
    acc_x = data.get('acc_x', 0)
    acc_y = data.get('acc_y', 0)
    acc_z = data.get('acc_z', 0)
    heart_rate = data.get('heart_rate', 0)
    # Dummy stress calculation
    stress_level = heart_rate / 100 + (abs(acc_x) + abs(acc_y) + abs(acc_z)) / 30
    return jsonify({
        'status': 'success',
        'stress_level': round(stress_level, 2)
    })


# --- Simulated endpoint for dashboard ---
import numpy as np
@app.route('/latest-result', methods=['GET'])
def latest_result():
    features = {
        "mean_hr": round(np.random.uniform(60, 110), 1),
        "sdnn": round(np.random.uniform(20, 80), 2),
        "rmssd": round(np.random.uniform(15, 60), 2),
        "acc_mean": round(np.random.uniform(0, 15), 3),
        "temp_mean": round(np.random.uniform(30, 37), 2),
    }
    prediction = int(features["mean_hr"] > 90 or features["acc_mean"] > 10 or features["temp_mean"] < 32)
    return jsonify({"features": features, "prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
