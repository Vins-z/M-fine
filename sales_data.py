import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 10000

# Generate dates
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i % 365) for i in range(n_samples)]  # Cycling through a year

# Define occupation categories and their corresponding income ranges
occupations = {
    'Student': (15000, 30000),
    'Teacher': (40000, 80000),
    'Engineer': (60000, 150000),
    'Doctor': (100000, 300000),
    'Lawyer': (80000, 250000),
    'Salesperson': (30000, 100000),
    'Manager': (70000, 200000),
    'Artist': (20000, 80000),
    'Entrepreneur': (50000, 500000),
    'Accountant': (50000, 120000)
}

# Generate occupation and income data
occupation_choices = list(occupations.keys())
occupation_data = np.random.choice(occupation_choices, n_samples)
income_data = [np.random.uniform(*occupations[occ]) for occ in occupation_data]

# Generate other columns
data = {
    'ORDERDATE': dates,
    'PRICE': np.random.uniform(10, 1000, n_samples).round(2),
    'SALES': np.random.uniform(100, 10000, n_samples).round(2),
    'CITY': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], n_samples),
    'COUNTRY': np.random.choice(['USA', 'Canada', 'Mexico', 'UK', 'Germany'], n_samples),
    'PRODUCTLINE': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Toys', 'Food & Beverage', 'Automotive', 'Beauty', 'Jewelry'], n_samples),
    'CUSTOMERNAME': [f'Customer_{i}' for i in range(1, n_samples + 1)],
    'DEALSIZE': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
    'OCCUPATION': occupation_data,
    'ANNUAL_INCOME': income_data
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some correlations and complexity
df['SALES'] = df['PRICE'] * np.random.uniform(1, 10, n_samples) + np.random.normal(0, 100, n_samples)
df['SALES'] *= np.where(df['DEALSIZE'] == 'Large', 1.2, np.where(df['DEALSIZE'] == 'Medium', 1.0, 0.8))
df['SALES'] *= np.where(df['COUNTRY'] == 'USA', 1.1, np.where(df['COUNTRY'] == 'Canada', 1.0, 0.9))

# Adjust sales based on income (higher income, slightly higher sales)
income_factor = (df['ANNUAL_INCOME'] / df['ANNUAL_INCOME'].mean()) ** 0.5
df['SALES'] *= income_factor

# Ensure sales are positive
df['SALES'] = np.maximum(df['SALES'], 10)

# Add a seasonal component to sales
df['MONTH'] = pd.to_datetime(df['ORDERDATE']).dt.month
seasonal_factor = np.sin(df['MONTH'] * np.pi / 6) * 0.2 + 1  # Peak in summer, trough in winter
df['SALES'] *= seasonal_factor

# Round numerical columns
df['SALES'] = df['SALES'].round(2)
df['ANNUAL_INCOME'] = df['ANNUAL_INCOME'].round(2)

# Save to CSV
df.to_csv('sample_sales_data_10k_with_income.csv', index=False)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())
print("\nOccupation distribution:")
print(df['OCCUPATION'].value_counts(normalize=True))