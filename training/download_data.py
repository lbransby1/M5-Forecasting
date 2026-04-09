import os
import pandas as pd
import requests
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
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            pbar.update(len(data))
    return file_path

def setup_raw_data():
    RAW_DATA_DIR = "../data/raw"
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    urls = [
        "https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_evaluation.csv",
        "https://huggingface.co/datasets/kashif/M5/resolve/main/calendar.csv",
        "https://huggingface.co/datasets/kashif/M5/resolve/main/sell_prices.csv"
    ]
    
    for url in urls:
        download_file(url, RAW_DATA_DIR)
    print("✅ All raw files secured in data/raw")

if __name__ == "__main__":
    setup_raw_data()