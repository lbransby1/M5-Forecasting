import os
import requests
import argparse
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest_folder):
    filename = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)
    
    if os.path.exists(file_path):
        print(f"--- {filename} already exists. Skipping. ---")
        return file_path

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(file_path, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=filename
    ) as pbar:
        for data in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks for speed
            if data:
                f.write(data)
                pbar.update(len(data))
    return file_path

def setup_raw_data(dest_dir):
    # Ensure dest_dir is a Path object for safety
    RAW_DATA_DIR = Path(dest_dir) 
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    urls = [
        "https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_evaluation.csv",
        "https://huggingface.co/datasets/kashif/M5/resolve/main/calendar.csv",
        "https://huggingface.co/datasets/kashif/M5/resolve/main/sell_prices.csv"
    ]
    
    for url in urls:
        download_file(url, RAW_DATA_DIR)
    print(f"✅ All raw files secured in {RAW_DATA_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This matches the --output_dir flag in your run_pipeline.sh
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()
    
    setup_raw_data(args.output_dir)