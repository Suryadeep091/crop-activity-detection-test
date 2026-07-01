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
# ... after COPY . . ...
# Explicitly force-copy the master file to prevent any ignore-rules from dropping it
COPY Telangana_Tehsil_Master.csv /app/Telangana_Tehsil_Master.csv

# 4. CRITICAL: Add /app to the Python path so local imports work
ENV PYTHONPATH=/app

# Start the FastAPI backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]







# curl -X POST "https://terradristi-crop-activity-413500342905.asia-south1.run.app/analyze/summary" \
# -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
# -H "Content-Type: application/json" \
# -d '{
#   "task_id": "KH_5_20260304_1719",
#   "coords": [
#     [78.672935118, 19.575615155], [78.672795142, 19.575659114], [78.67164674, 19.576093755], 
#     [78.671501669, 19.576130897], [78.671551117, 19.576286233], [78.671587835, 19.576404242], 
#     [78.671615668, 19.576519867], [78.671652879, 19.576642059], [78.67168153, 19.577398296], 
#     [78.671693085, 19.57753087], [78.671830216, 19.577498041], [78.672526118, 19.577250474], 
#     [78.673129045, 19.577512695], [78.673462737, 19.577684458], [78.673638233, 19.577723399], 
#     [78.673571599, 19.575615155], [78.672929556, 19.575756022], [78.672935118, 19.575615155]
#   ],
#   "end_date": "2026-03-04",
#   "properties": {
#     "State": "telangana",
#     "District": "Adilabad",
#     "Tehsil": "Adilabad Rural",
#     "Village": "Khandala",
#     "Khasra_No": "5",
#     "Area_ac": "7.39",
#     "ownership": "None"
#   }
# }' -o report.pdf