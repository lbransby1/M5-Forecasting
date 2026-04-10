FROM python:3.11-slim

# 1. Install system dependencies for LightGBM (CRITICAL FIX)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the project
COPY . .

# 4. Make sure your scripts are executable
RUN chmod +x start.sh scripts/run_pipeline.sh

# 5. EXPOSE the port (Railway uses $PORT, but we'll list 8080 as a default)
EXPOSE 8080

# 6. Run the start script which triggers the pipeline then the API
CMD ["./start.sh"]