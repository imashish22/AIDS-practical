import pandas as pd
from scipy import stats
df = pd.read_csv('realestate3.csv')
# Calculate z-scores for numerical columns
z_scores = stats.zscore(df[['bed', 'bath', 'acre_lot', 'house_size', 'price']])

# Define threshold for identifying outliers
threshold = 3

# Find indices of outliers
outlier_indices = (z_scores > threshold).any(axis=1)

# Remove outliers from the dataset
df_no_outliers = df[~outlier_indices]


# Save the filled dataset to a new CSV file named 'realestate2.csv'
df_no_outliers.to_csv('realestate4.csv', index=False)
