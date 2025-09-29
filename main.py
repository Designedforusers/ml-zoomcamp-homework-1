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

# Question 3: Sum of weights calculation
print("\n" + "="*50)
print("Q3: Sum of weights calculation")
print("="*50)

# Step 1: Select all cars from Asia
asian_cars_q3 = df[df['origin'] == 'Asia']
print(f"Number of cars from Asia: {len(asian_cars_q3)}")

# Step 2: Select only columns vehicle_weight and model_year
selected_cols = asian_cars_q3[['vehicle_weight', 'model_year']]
print(f"Selected columns shape: {selected_cols.shape}")

# Step 3: Select the first 7 values
first_7 = selected_cols.head(7)
print("First 7 values:")
print(first_7)

# Step 4: Get the underlying NumPy array (X)
X = first_7.values
print(f"\nX shape: {X.shape}")
print("X array:")
print(X)

# Step 5: Compute X.T @ X (matrix multiplication)
XTX = X.T @ X
print(f"\nXTX shape: {XTX.shape}")
print("XTX matrix:")
print(XTX)

# Step 6: Invert XTX
XTX_inv = np.linalg.inv(XTX)
print(f"\nInverse of XTX:")
print(XTX_inv)

# Step 7: Create array y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
print(f"\ny array: {y}")

# Step 8: Multiply: inverse(XTX) @ X.T @ y = w
w = XTX_inv @ X.T @ y
print(f"\nw result: {w}")

# Step 9: Sum of all elements of w
w_sum = np.sum(w)
print(f"\nSum of all elements of w: {w_sum}")

print("\nDataset loaded successfully! Ready for machine learning analysis.")