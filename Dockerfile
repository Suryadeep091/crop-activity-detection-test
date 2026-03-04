FROM python:3.10-slim

# 1. Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Upgrade pip to ensure the latest build logic is used
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install --with-deps chromium
# 3. Copy all project files (including location.py, etc.)
COPY . .

# 4. CRITICAL: Add /app to the Python path so local imports work
ENV PYTHONPATH=/app

# Start the FastAPI backend
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]