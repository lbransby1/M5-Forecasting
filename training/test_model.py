import sys
import os
import argparse
import polars as pl

# 1. Path routing for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from core import feature_engineering
from core import pre_process

pre_process.preprocess_m5(mode="local", raw_dir="data/raw", output_path="data/processed/m5-processed.parquet", add_features=True)

