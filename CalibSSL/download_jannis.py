import pandas as pd
from sklearn.datasets import fetch_openml
import os

def download_dataset():
    os.makedirs('data/data/raw', exist_ok=True)
    print("Downloading Jannis dataset (83,733 rows)...")
    data = fetch_openml(data_id=41168, as_frame=True, parser='auto')
    data.frame.to_csv('data/data/raw/jannis.csv', index=False)
    print("Downloaded and saved exactly to data/data/raw/jannis.csv!")

if __name__ == "__main__":
    download_dataset()
