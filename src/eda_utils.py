# src/eda_utils.py
import pandas as pd
import numpy as np

def load_insurance_data(file_path):
    """
    Loads the AlphaCare insurance text data using the pipe (|) delimiter
    and cleans up any broken column header fragments.
    """
    try:
        # Step 1: Read the file explicitly using pipe separation
        df = pd.read_csv(file_path, sep='|', engine='python')
        
        # Step 2: Clean the column headers (strip spaces or weird characters)
        df.columns = [col.strip() for col in df.columns]
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_loss_ratio(df, claims_col='TotalClaims', premium_col='TotalPremium'):
    """Loss Ratio = TotalClaims / TotalPremium"""
    # Ensure columns are numeric before summing
    total_claims = pd.to_numeric(df[claims_col], errors='coerce').sum()
    total_premium = pd.to_numeric(df[premium_col], errors='coerce').sum()
    return total_claims / total_premium if total_premium != 0 else 0