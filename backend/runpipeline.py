import subprocess
import sys
import os

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    # Using sys.executable ensures it uses the same python environment
    result = subprocess.run([sys.executable, script_name], check=True)
    if result.returncode != 0:
        print(f"❌ Error: {script_name} failed. Halting pipeline.")
        sys.exit(1)
    print(f"✅ {script_name} completed successfully.\n")

if __name__ == "__main__":
    try:
        # 0. Upload Data to S3
        run_script("upload_data.py")
        
        # 1. Data Preparation
        run_script("pre-process.py")
        
        # 2. Product Naming
        run_script("name-map.py")
        
        # 3. Model Training (Note: This may take a while)
        run_script("train.py")
        
        # 4. Start the FastAPI Server
        print("🚀 All systems ready. Starting FastAPI Backend...")
        # We use subprocess.Popen for the server so it stays running
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

    except subprocess.CalledProcessError as e:
        print(f"\n🛑 Pipeline interrupted: {e}")
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")