import pandas as pd
import numpy as np

# Load the car fuel efficiency dataset
print("Loading car fuel efficiency dataset...")
df = pd.read_csv('car_fuel_efficiency.csv')

# Display basic information about the dataset
print("\nDataset Info:")
print(f"Shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset description:")
print(df.describe())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset loaded successfully! Ready for machine learning analysis.")