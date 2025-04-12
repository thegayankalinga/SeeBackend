import os
import json
import gdown

MODELS_PATH = "/app/models"
VERSION_FILE = os.path.join(MODELS_PATH, "deployed_versions.json")

# Load desired model versions
with open("model_versions.json", "r") as f:
    model_versions = json.load(f)

# Load deployed versions (if exists)
if os.path.exists(VERSION_FILE):
    with open(VERSION_FILE, "r") as f:
        deployed_versions = json.load(f)
else:
    deployed_versions = {}

# Ensure models directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

for model_name, info in model_versions.items():
    filename = info["filename"]
    file_id = info["id"]
    dest_path = os.path.join(MODELS_PATH, filename)

    # Check if we already have this version
    if deployed_versions.get(model_name) == filename and os.path.exists(dest_path):
        print(f"✅ {model_name} ({filename}) already present — skipping.")
        continue

    # Otherwise download the model
    print(f"⬇️ Downloading {model_name} → {filename} ...")
    gdown.download(id=file_id, output=dest_path, quiet=False)

    # Update the deployed version
    deployed_versions[model_name] = filename

# Save updated version info
with open(VERSION_FILE, "w") as f:
    json.dump(deployed_versions, f, indent=2)

print("✅ Model versioning complete.")