FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV-
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies (add fastapi and uvicorn if not in requirements)
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi==0.109.2 \
    uvicorn==0.27.1 \
    python-multipart==0.0.9

COPY . .

# Ensure checkpoints directory exists
RUN mkdir -p checkpoints

EXPOSE 8000 8501

# The entrypoint will be overridden by docker-compose for api vs web
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
