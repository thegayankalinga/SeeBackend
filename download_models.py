import os
import gdown

MODELS_PATH = "/app/models"
os.makedirs(MODELS_PATH, exist_ok=True)

MODEL_FILES = {
    "best_hybrid_model.pkl": "13e3VU6hlLh6j70ux0rHtJf7tgKfyKcN1",
    "best_random_forest.pkl": "13bZ-O2Ic1KZJy0xrLFKCTtrxkZTqnv1w",
    "best_xgboost.pkl": "13fFsizsQbpbIuiNBb7Clacc3TvKs4aFq",
    "lstm_model.keras": "13gXp0PhnDb__PQEnG8mUkafvmM-uoJSK",
    "mlp_model.keras": "13fexWW4zG9X9p7dty5FNb-sEyAaMPO7c"
}

for filename, file_id in MODEL_FILES.items():
    dest_path = os.path.join(MODELS_PATH, filename)
    if not os.path.exists(dest_path):
        print(f"⬇️ Downloading {filename}...")
        gdown.download(id=file_id, output=dest_path, quiet=False)
    else:
        print(f"✅ {filename} already exists — skipping.")