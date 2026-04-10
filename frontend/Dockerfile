FROM python:3.11-slim

WORKDIR /app

# 1. Install dependencies from the frontend subfolder
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy everything from the root so it can see the frontend folder
COPY . .

# 3. Expose the port
EXPOSE 8501

# 4. Use the full path to the module to ensure it starts
CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]