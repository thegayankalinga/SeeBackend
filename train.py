import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from azure.storage.blob import BlobServiceClient
import os


def train_model(data, target):
    # Your model training code here
    model = YourModel()
    model.fit(data, target)

    # Save locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}.pkl"
    local_path = os.path.join("models", model_filename)

    with open(local_path, 'wb') as f:
        pickle.dump(model, f)

    # Upload to Azure Blob Storage
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_name = "models"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_filename)

    with open(local_path, "rb") as data:
        blob_client.upload_blob(data)

    return model_filename


if __name__ == "__main__":
    # Add your training data loading and preprocessing here
    trained_model = train_model(X, y)