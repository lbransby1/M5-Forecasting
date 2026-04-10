FROM python:3.11-slim

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install requirements (from the backend folder)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the backend code into the container's backend folder
COPY backend/ ./backend/

# 3. Copy the models from the ROOT models folder into the container
COPY models/ ./models/

# 4. Copy the start script from the root
COPY start.sh .
RUN chmod +x start.sh

# 5. Set Python path so it can find the backend module
ENV PYTHONPATH=/app

CMD ["./start.sh"]