import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X = pd.read_csv("data/training_data.csv")
Y = pd.read_csv("data/training_data_targets.csv", header=None)

print("\nShape of training data: ", X.shape)
print("\nShape of training target data: ", Y.shape)

# Check for missing values
missing_values = X.isnull().sum()

missing_values_count = missing_values[missing_values > 0]
total_missing_values = missing_values_count.sum()
print("\n\n Number of missing values for each column:")
print(missing_values_count)
print("\nTotal Missing Values: ", total_missing_values)

# Plot a bar chart for missing values
plt.figure(figsize=(12, 6))
missing_values.plot(kind='bar')
plt.title('Number of Missing Values for Each Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.show()

# Calculate correlation matrix
correlation_matrix = X.corr()

# Plot heatmap
plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.7)
plt.title("Correlation Matrix")
plt.show()

# Filter values with correlation > 0.6 and exclude same feature pairs
high_correlation = correlation_matrix[(correlation_matrix > 0.6) & (correlation_matrix < 1)]

# Get the indices of pairs with correlation > 0.6
high_corr_indices = [(i, j) for i in range(correlation_matrix.shape[0]) for j in range(i+1, correlation_matrix.shape[1]) if high_correlation.iloc[i, j] > 0.6]

print("\n\nPairs with correlation > 0.6:")
for i, j in high_corr_indices:
    print(f"{X.columns[i]} - {X.columns[j]}: {high_correlation.iloc[i, j]}")

class_counts = Y.value_counts()
print("\n\nNumber of samples in each class:")
print(class_counts)


