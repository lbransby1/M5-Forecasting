import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import os

endpoint = os.getenv("STORAGE_ENDPOINT_URL")
access_key = os.getenv("STORAGE_ACCESS_KEY_ID")
secret_key = os.getenv("STORAGE_SECRET_ACCESS_KEY")


def upload_to_s3(file_path, bucket_name, s3_key):
    """Upload a file to an S3 bucket."""
    s3_client = boto3.client('s3') # Ensure AWS_ACCESS_KEY_ID etc. are in your ENV
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        
        try:
            print(f"⬆Uploading {file_path} to s3://{bucket_name}/{s3_key}...")
            s3_client.upload_file(str(file_path), bucket_name, s3_key)
            print("Upload successful.")
        except ClientError as e:
            print(f"Upload failed: {e}")
    else:
        print(f"File already exists: {file_path}")

if __name__ == "__main__":
    upload_to_s3("backend/data/raw/calendar.csv", "m5-walmart-data", "calendar.csv")
    upload_to_s3("backend/data/raw/sell_prices.csv", "m5-walmart-data", "sell_prices.csv")
    upload_to_s3("backend/data/raw/sales_train_evaluation.csv", "m5-walmart-data", "sales_train_evaluation.csv")
    upload_to_s3("backend/data/raw/sales_train_validation.csv", "m5-walmart-data", "sales_train_validation.csv")
    upload_to_s3("backend/data/raw/sales_train_validation.csv", "m5-walmart-data", "sales_train_validation.csv")