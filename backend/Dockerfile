FROM python:3.11-slim

# 1. Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install requirements
# Since your requirements are at the root, we copy from root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy everything from the root
COPY . .

# 4. CRITICAL: Give permission to the script and ensure the port matches Railway
RUN chmod +x start.sh scripts/run_pipeline.sh
EXPOSE 8080

# 5. Run the script instead of uvicorn
CMD ["./start.sh"]