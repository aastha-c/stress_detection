import qrcode
import socket

# Get local IP address
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('10.255.255.255', 1))
    local_ip = s.getsockname()[0]
except Exception:
    local_ip = '127.0.0.1'
finally:
    s.close()

# URL to access sensor_form.html via Python's http.server
backend_url = f"http://{local_ip}:8000/sensor_form.html"

img = qrcode.make(backend_url)
img.save("sensor_form_qr.png")
print(f"QR code saved as sensor_form_qr.png. URL: {backend_url}")
