import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import os

def download_all_datasets():
    """Download and save all 5 datasets"""
    
    # Create output directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # ============ 1. ADULT INCOME ============
    print("Downloading Adult Income...")
    adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    adult_df = adult.frame
    adult_df.to_csv('data/raw/adult.csv', index=False)
    print(f"Adult: {adult_df.shape}")
    
    # ============ 2. BANK MARKETING ============
    print("Downloading Bank Marketing...")
    bank = fetch_openml('bank-marketing', version=1, as_frame=True, parser='auto')
    bank_df = bank.frame
    bank_df.to_csv('data/raw/bank.csv', index=False)
    print(f"Bank: {bank_df.shape}")
    
    # ============ 3. CREDIT DEFAULT ============
    print("Downloading Credit Default...")
    # Using OpenML default of credit card clients
    credit = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
    credit_df = credit.frame
    credit_df.to_csv('data/raw/credit.csv', index=False)
    print(f"Credit: {credit_df.shape}")
    
    # ============ 4. COVERTYPE ============
    print("Downloading Covertype (this is large, ~580K samples)...")
    cover = fetch_openml('covertype', version=3, as_frame=True, parser='auto')
    cover_df = cover.frame
    cover_df.to_csv('data/raw/covertype.csv', index=False)
    print(f"Covertype: {cover_df.shape}")
    
    # ============ 5. DIABETES READMISSION ============
    print("Downloading Diabetes...")
    # Using OpenML diabetes dataset
    diabetes = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
    diabetes_df = diabetes.frame
    diabetes_df.to_csv('data/raw/diabetes.csv', index=False)
    print(f"Diabetes: {diabetes_df.shape}")
    
    print("\n✅ All datasets downloaded successfully!")
    return True

if __name__ == "__main__":
    download_all_datasets()