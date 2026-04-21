import urllib.request
import os

models_dir = "static/models"
base_url = "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/"

files = [
    "tiny_face_detector_model-weights_manifest.json",
    "tiny_face_detector_model-shard1",
    "face_expression_model-weights_manifest.json",
    "face_expression_model-shard1"
]

for file in files:
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(base_url + file, os.path.join(models_dir, file))
    
print("All models downloaded successfully!")
