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

print("\nFuel types analysis:")
print(f"Number of unique fuel types: {df['fuel_type'].nunique()}")
print("Fuel types present:")
print(df['fuel_type'].value_counts())

# Question 1: Maximum fuel efficiency of cars from Asia
print("\n" + "="*50)
print("Q1: Maximum fuel efficiency of cars from Asia")
print("="*50)
print("Origins in dataset:")
print(df['origin'].value_counts())
asian_cars = df[df['origin'] == 'Asia']
max_efficiency_asia = asian_cars['fuel_efficiency_mpg'].max()
print(f"\nMaximum fuel efficiency of cars from Asia: {max_efficiency_asia:.6f} mpg")

# Question 2: Horsepower median analysis
print("\n" + "="*50)
print("Q2: Horsepower median analysis")
print("="*50)

# Step 1: Find median value of horsepower
original_median = df['horsepower'].median()
print(f"Original median horsepower: {original_median}")

# Step 2: Find most frequent value of horsepower
most_frequent_hp = df['horsepower'].mode()[0]
print(f"Most frequent horsepower value: {most_frequent_hp}")

# Step 3: Fill missing values with most frequent value
df_filled = df.copy()
df_filled['horsepower'] = df_filled['horsepower'].fillna(most_frequent_hp)

# Step 4: Calculate median again
new_median = df_filled['horsepower'].median()
print(f"New median horsepower after filling: {new_median}")

# Step 5: Check if it changed
print(f"\nComparison:")
print(f"Original median: {original_median}")
print(f"New median: {new_median}")
if new_median > original_median:
    print("Answer: Yes, it increased")
elif new_median < original_median:
    print("Answer: Yes, it decreased")
else:
    print("Answer: No")

print("\nDataset loaded successfully! Ready for machine learning analysis.")