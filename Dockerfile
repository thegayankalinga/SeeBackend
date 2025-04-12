# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=3000 --no-cache-dir tensorflow==2.17.0
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt
RUN pip install gdown

# Copy all source files (main.py, download_models.py, etc.)
COPY . /app/

# Download models if not present
RUN python download_models.py

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]