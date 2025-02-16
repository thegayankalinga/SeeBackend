from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import sklearn
from pydantic import BaseModel, validator
import numpy as np
import joblib
import os
from typing import List, Dict, Optional, Union
import pandas as pd
from functools import lru_cache
from pipeline_imp import ProjectEffortPipeline  # Ensure this is correctly imported
import sys


class Settings:
    MODELS_PATH: str = os.getenv(
        "MODELS_PATH",
        "/Users/gayan/Library/CloudStorage/GoogleDrive-bg15407@gmail.com/My Drive/Projects/msc_project/models"
    )
    PIPELINE_PATH: str = os.getenv(
        "PIPELINE_PATH",
        "/Users/gayan/Library/CloudStorage/GoogleDrive-bg15407@gmail.com/My Drive/Projects/msc_project/pipelines/project_effort_pipeline.joblib"
    )
    DATA_PATH: str = os.getenv(
        "DATA_PATH",
        "/Users/gayan/Library/CloudStorage/GoogleDrive-bg15407@gmail.com/My Drive/Projects/msc_project/data_sets/project_mandays_calculations50k_augmented.csv"
    )
    MODEL_FILES = {
        "Hybrid": "best_hybrid_model.pkl",
        "RandomForest": "best_random_forest.pkl",
        "XGBoost": "best_xgboost.pkl",
        "LSTM": "lstm_model.keras",
        "MLP": "mlp_model.keras"
    }


settings = Settings()


#Initiate the Pipeline when running for the first time.
# pipeline = ProjectEffortPipeline()
#
# # Train the pipeline
# pipeline.fit(pd.read_csv(settings.DATA_PATH))
# # Saving the Pipeline
# joblib.dump(pipeline, settings.PIPELINE_PATH)

# print("✅ Pipeline saved successfully!")

app = FastAPI(
    title="Effort Estimation API",
    description="ML-powered API for software project effort estimation.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_tensorflow_compatibility():
    """Check TensorFlow version and compatibility."""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {tf.keras.__version__}")

        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Available GPUs: {gpus}")

        return True, tf.__version__
    except Exception as e:
        print(f"TensorFlow compatibility check failed: {str(e)}")
        return False, None
def load_tensorflow():
    """Lazy loading of TensorFlow"""
    try:
        import tensorflow as tf
        return tf, True
    except Exception as e:
        print(f"TensorFlow import failed: {str(e)}")
        return None, False


@lru_cache()
def get_pipeline():
    try:
        print(f"Loading pipeline from: {settings.PIPELINE_PATH}")

        pipeline = joblib.load(settings.PIPELINE_PATH)

        print("✅ Pipeline Loaded Successfully")
        # print(f"🔍 Loaded Object Type: {type(pipeline)}")

        return pipeline
    except Exception as e:
        print(f"❌ Failed to load pipeline: {str(e)}")
        raise RuntimeError(f"Failed to load pipeline: {str(e)}")


class PredictionRequest(BaseModel):
    features: List[Union[str, int, float]]
    model_name: str

    @validator("model_name")
    def validate_model_name(cls, v):
        valid_models = ["Hybrid", "RandomForest", "XGBoost", "LSTM", "MLP"]
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        if v in ["LSTM", "MLP"]:
            tf, available = load_tensorflow()
            if not available:
                raise ValueError("TensorFlow models (LSTM, MLP) are not available")
        return v


class PredictionResponse(BaseModel):
    success: bool
    model: str
    predictions: Dict[str, float]


#Load the Hybrid Models
def load_hybrid_model(settings):
    """Loads the Hybrid Model (LSTM + XGBoost) and prepares it for prediction."""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    import joblib
    import os

    # Create the LSTM model architecture first
    input_shape = (1, 41)  # (timesteps, features)

    # Create input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Add LSTM layers (matching your training architecture)
    lstm_1 = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)(inputs)
    lstm_2 = tf.keras.layers.LSTM(64, activation='relu')(lstm_1)
    outputs = tf.keras.layers.Dense(4, activation='linear')(lstm_2)

    # Create the full model
    full_model = Model(inputs=inputs, outputs=outputs)

    # Load the weights from your saved model
    full_model.load_weights(os.path.join(settings.MODELS_PATH, "lstm_model.keras"))

    # Create feature extractor model using the second LSTM layer
    lstm_feature_extractor = Model(inputs=full_model.input, outputs=full_model.layers[2].output)

    # Load XGBoost Model
    xgb_model_path = os.path.join(settings.MODELS_PATH, "best_hybrid_model.pkl")
    xgb_model = joblib.load(xgb_model_path)

    return lstm_feature_extractor, xgb_model


def predict_hybrid(features, lstm_feature_extractor, xgb_model):
    """Generate predictions using the hybrid model."""
    import numpy as np

    # Reshape features for LSTM (assuming features is already normalized/processed)
    lstm_input = features.values.reshape(1, 1, -1)

    # Extract LSTM features
    lstm_features = lstm_feature_extractor.predict(lstm_input, verbose=0)

    # Ensure features are 2D
    if len(lstm_features.shape) == 3:
        lstm_features = lstm_features.reshape(lstm_features.shape[0], -1)

    # Combine original features with LSTM features
    combined_features = np.concatenate((features.values, lstm_features), axis=1)

    # Generate predictions using XGBoost
    predictions = xgb_model.predict(combined_features)

    return predictions



#Load the LSTM Model
def load_lstm_model(model_path: str):
    """Specialized function to load LSTM models."""
    tf, available = load_tensorflow()
    if not available:
        raise RuntimeError("TensorFlow is not available")

    print(f"TensorFlow version: {tf.__version__}")
    tf.keras.backend.clear_session()

    # Load as tf.saved_model format
    try:
        model = tf.saved_model.load(model_path)
        print("Successfully loaded LSTM model using saved_model format")
        return model
    except Exception as e:
        print(f"saved_model loading failed, trying HDF5: {str(e)}")
        # Fallback to HDF5 format
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=None,
            compile=False
        )
        print("Successfully loaded LSTM model using HDF5 format")
        return model

#Load Model in general
@lru_cache()
def load_model(model_name: str):
    """Load and cache ML models."""
    try:
        if model_name not in settings.MODEL_FILES:
            raise ValueError(f"Invalid model name: {model_name}")

        model_filename = settings.MODEL_FILES[model_name]
        model_path = os.path.join(settings.MODELS_PATH, model_filename)

        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        if model_name == "LSTM":
            return load_lstm_model(model_path)
        elif model_name == "MLP":
            tf, available = load_tensorflow()
            if not available:
                raise RuntimeError("TensorFlow is not available")

            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=None
            )
            model.trainable = False
            return model
        else:
            # For Hybrid, RandomForest, and XGBoost models
            return joblib.load(model_path)

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")


def convert_feature_values(value: Union[str, int, float]) -> Union[float, str]:
    """Convert feature values to appropriate types."""
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        if value.lower() in ['yes', 'no']:
            return value.lower()
        try:
            return float(value)
        except ValueError:
            return value

    return str(value)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status."""
    tf, tensorflow_available = load_tensorflow()
    available_models = ["Hybrid", "RandomForest", "XGBoost"]
    if tensorflow_available:
        available_models.extend(["LSTM", "MLP"])

    return {
        "status": "healthy",
        "version": app.version,
        "available_models": available_models,
        "tensorflow_available": tensorflow_available
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_effort(request: PredictionRequest):
    """
    Generate effort predictions using the specified model.
    """
    try:

        pipeline = get_pipeline()


        # Convert string values to appropriate types
        processed_features = [convert_feature_values(val) for val in request.features]

        # Transform input features using pipeline
        transformed_data = pipeline.transform(processed_features)

        #New Hybrid Code
        if request.model_name == "Hybrid":
            # Load hybrid model components
            lstm_feature_extractor, xgb_model = load_hybrid_model(settings)

            # Generate predictions using hybrid approach
            predictions = predict_hybrid(transformed_data, lstm_feature_extractor, xgb_model)
        else:
            # Handle other models as before
            model = load_model(request.model_name)

            if request.model_name == "LSTM":
                input_data = transformed_data.values.reshape(1, 1, -1)
                predictions = model.predict(input_data, verbose=0)
            else:
                predictions = model.predict(transformed_data)


        # Load model and generate predictions
        # model = load_model(request.model_name)
        #
        # # Reshape input for LSTM model
        # if request.model_name == "LSTM":
        #     input_data = transformed_data.values.reshape(1, 1, -1)
        #     predictions = model.predict(input_data, verbose=0)
        # else:
        #     predictions = model.predict(transformed_data)

        # Inverse transform the predictions
        original_scale_predictions = pipeline.inverse_transform_targets(predictions)

        # Format output
        effort_categories = [
            "Delivery Effort",
            "Engineering Effort",
            "DevOps Effort",
            "QA Effort"
        ]

        output = {
            category: round(float(value), 2)
            for category, value in zip(effort_categories, original_scale_predictions[0])
        }

        return PredictionResponse(
            success=True,
            model=request.model_name,
            predictions=output
        )

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


