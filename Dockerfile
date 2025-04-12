# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gdown



# Copy app code and model
COPY ./app /app/app

# Create models dir and download them
RUN python download_models.py

#COPY ./models /app/models
#COPY ./pipelines /app/pipelines
#COPY ./data_sets /app/data_sets

# Expose FastAPI port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]