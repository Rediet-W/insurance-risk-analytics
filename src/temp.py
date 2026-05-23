# run this in your terminal or a temporary python script/notebook cell
import pandas as pd

# Load the raw file
file_path = "data/MachineLearningRating_v3.txt"
df = pd.read_csv(file_path, sep='|', engine='python')

# Perform a basic data cleaning step: fill missing values in key columns
df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce').fillna(0)
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)

# Save it back to the EXACT same location to overwrite it
df.to_csv(file_path, sep='|', index=False)
print("Data cleaning complete! Version 2 generated.")