"""
Live Stress Detection Server
==============================
FastAPI server with WebSocket support for real-time phone sensor
streaming and dashboard updates.

Endpoints:
    GET  /                → Serves the live dashboard
    GET  /phone           → Serves the phone sensor client
    GET  /qr              → Returns a QR code image for phone connection
    WS   /ws/sensor       → Phone sends sensor data here
    WS   /ws/dashboard    → Dashboard receives predictions here

Run:
    python live_server.py
"""

import os
import sys
import io
import json
import time
import asyncio
import socket
import numpy as np
from collections import deque
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.motion_features import extract_motion_features
import joblib

# ── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="Stress Detection Live Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    print(f"  [REQ] {request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms)")
    return response

# Absolute path for static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Health Check ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Model Loading ───────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "motion_stress_model.pkl")
clf = None
feature_names = None


def load_model():
    global clf, feature_names
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        clf = saved["model"]
        feature_names = saved["feature_names"]
        print(f"  [OK] Loaded motion stress model ({len(feature_names)} features)")
    else:
        print(f"  [WARN] No model found at {MODEL_PATH}")
        print(f"    Run: python model/train_motion_model.py")


# ── Sensor Data Buffer ─────────────────────────────────────────────────────
BUFFER_SIZE = 200  # ~10 seconds at 20Hz
WINDOW_SIZE = 100  # ~5 seconds of data for feature extraction

sensor_buffer = {
    "acc_x": deque(maxlen=BUFFER_SIZE),
    "acc_y": deque(maxlen=BUFFER_SIZE),
    "acc_z": deque(maxlen=BUFFER_SIZE),
    "gyro_alpha": deque(maxlen=BUFFER_SIZE),
    "gyro_beta": deque(maxlen=BUFFER_SIZE),
    "gyro_gamma": deque(maxlen=BUFFER_SIZE),
    "timestamps": deque(maxlen=BUFFER_SIZE),
}

latest_prediction = {
    "stress_probability": 0.0,
    "prediction": 0,
    "label": "Relaxed",
    "features": {},
    "timestamp": 0,
    "sensor_count": 0,
    "connected": False,
}

# Connected dashboard WebSockets
dashboard_clients: List[WebSocket] = []
phone_connected = False


def get_local_ip() -> str:
    """Get the local IP address for the phone to connect to."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


latest_phone_face_stress = 0.0

# ── Feature Extraction & Prediction ────────────────────────────────────────
def predict_stress():
    """Extract features from the sensor buffer and predict stress."""
    global latest_prediction

    if clf is None or feature_names is None:
        return

    if len(sensor_buffer["acc_x"]) < WINDOW_SIZE:
        return

    # Get the latest window of data
    acc_x = np.array(list(sensor_buffer["acc_x"]))[-WINDOW_SIZE:]
    acc_y = np.array(list(sensor_buffer["acc_y"]))[-WINDOW_SIZE:]
    acc_z = np.array(list(sensor_buffer["acc_z"]))[-WINDOW_SIZE:]
    gyro_a = np.array(list(sensor_buffer["gyro_alpha"]))[-WINDOW_SIZE:]
    gyro_b = np.array(list(sensor_buffer["gyro_beta"]))[-WINDOW_SIZE:]
    gyro_g = np.array(list(sensor_buffer["gyro_gamma"]))[-WINDOW_SIZE:]

    # Extract features
    features = extract_motion_features(
        acc_x, acc_y, acc_z,
        gyro_a, gyro_b, gyro_g,
        fs=20.0,
    )

    # Build feature vector in correct order
    x = np.array([[features.get(name, 0.0) for name in feature_names]])
    x = np.nan_to_num(x, nan=0.0)

    # Predict
    prediction = int(clf.predict(x)[0])
    probability = clf.predict_proba(x)[0]
    stress_prob = float(probability[1])

    latest_prediction = {
        "stress_probability": round(stress_prob, 4),
        "prediction": prediction,
        "label": "Stressed" if prediction == 1 else "Relaxed",
        "features": {k: round(v, 4) for k, v in features.items()},
        "timestamp": time.time(),
        "sensor_count": len(sensor_buffer["acc_x"]),
        "connected": phone_connected,
        "phone_face_stress": latest_phone_face_stress,
    }


# ── WebSocket: Phone Sensor Input ──────────────────────────────────────────
@app.websocket("/ws/sensor")
async def sensor_websocket(websocket: WebSocket):
    global phone_connected, latest_phone_face_stress
    await websocket.accept()
    phone_connected = True
    print("  [PHONE] Phone sensor connected!")

    sample_count = 0
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            # Buffer the sensor readings
            sensor_buffer["acc_x"].append(msg.get("ax", 0))
            sensor_buffer["acc_y"].append(msg.get("ay", 0))
            sensor_buffer["acc_z"].append(msg.get("az", 9.81))
            sensor_buffer["gyro_alpha"].append(msg.get("ga", 0))
            sensor_buffer["gyro_beta"].append(msg.get("gb", 0))
            sensor_buffer["gyro_gamma"].append(msg.get("gg", 0))
            sensor_buffer["timestamps"].append(msg.get("t", time.time()))
            
            if "face_stress" in msg:
                latest_phone_face_stress = msg["face_stress"]

            sample_count += 1

            # Run prediction every 10 samples (~0.5 seconds)
            if sample_count % 10 == 0:
                predict_stress()

                # Send latest prediction back to phone
                await websocket.send_text(json.dumps(latest_prediction))

                # Broadcast to all dashboard clients
                await broadcast_to_dashboards()

    except WebSocketDisconnect:
        phone_connected = False
        latest_prediction["connected"] = False
        print("  [PHONE] Phone sensor disconnected")
        await broadcast_to_dashboards()


# ── WebSocket: Dashboard ───────────────────────────────────────────────────
@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.append(websocket)
    print(f"  [DASH] Dashboard connected (total: {len(dashboard_clients)})")

    try:
        # Send current state immediately
        await websocket.send_text(json.dumps(latest_prediction))

        # Keep connection alive
        while True:
            # Wait for any message (ping/pong keepalive)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text(json.dumps({"keepalive": True}))
    except WebSocketDisconnect:
        dashboard_clients.remove(websocket)
        print(f"  [DASH] Dashboard disconnected (total: {len(dashboard_clients)})")


async def broadcast_to_dashboards():
    """Send latest prediction to all connected dashboards."""
    if not dashboard_clients:
        return
    message = json.dumps(latest_prediction)
    disconnected = []
    for client in dashboard_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        dashboard_clients.remove(client)


# ── HTTP Endpoints ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the live dashboard page."""
    html_path = os.path.join(os.path.dirname(__file__), "live_dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/phone", response_class=HTMLResponse)
async def serve_phone():
    """Serve the phone sensor client page."""
    html_path = os.path.join(os.path.dirname(__file__), "phone_sensor.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def _get_base_url(request: Request) -> str:
    """Build the correct base URL for QR codes and phone connections.
    
    In production (Railway/Render), uses the public domain from env vars or Host header.
    In local development, falls back to LAN IP.
    """
    # Check for cloud platform environment variables
    railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    
    if railway_domain:
        return f"https://{railway_domain}"
    if render_url:
        return render_url.rstrip("/")
    
    # Check if the Host header looks like a public domain (not a local IP)
    host_header = request.headers.get("host", "")
    if host_header and not any(host_header.startswith(p) for p in ["127.", "192.168.", "10.", "172.", "localhost"]):
        scheme = request.headers.get("x-forwarded-proto", "https")
        return f"{scheme}://{host_header}"
    
    # Local development fallback — use LAN IP
    local_ip = get_local_ip()
    if ":" in host_header:
        port = host_header.split(":")[-1]
    else:
        port = str(os.environ.get("PORT", 8000))
    return f"https://{local_ip}:{port}"


@app.get("/qr")
async def generate_qr(request: Request):
    """Generate a QR code for the phone to scan."""
    try:
        import qrcode
    except ImportError:
        return JSONResponse({"error": "qrcode package not installed."}, status_code=500)

    base_url = _get_base_url(request)
    url = f"{base_url}/phone"

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a2e", back_color="#ffffff")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/status")
async def api_status(request: Request):
    """Get current system status."""
    base_url = _get_base_url(request)
    return {
        "model_loaded": clf is not None,
        "phone_connected": phone_connected,
        "buffer_size": len(sensor_buffer["acc_x"]),
        "latest_prediction": latest_prediction,
        "phone_url": f"{base_url}/phone",
        "dashboard_url": f"{base_url}/",
        "dashboard_clients": len(dashboard_clients),
    }


# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_model()
    print("\n" + "=" * 60)
    print("  Stress Detection Live System [SSL ENABLED]")
    print("=" * 60)
    print("\n  Server initialized and ready.")
    print("  Ensure you access via HTTPS.")
    print("  Connect your mobile device using the QR code on the dashboard.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # SSL Configuration
    ssl_key = "key.pem"
    ssl_cert = "cert.pem"
    
    if os.path.exists(ssl_key) and os.path.exists(ssl_cert):
        print(f"  [START] Starting server with SSL (HTTPS)")
        uvicorn.run(
            "live_server:app", 
            host=host, 
            port=port, 
            reload=False,
            ssl_keyfile=ssl_key,
            ssl_certfile=ssl_cert
        )
    else:
        print(f"  [WARN] SSL certificates not found. Starting on HTTP.")
        uvicorn.run("live_server:app", host=host, port=port, reload=False)
