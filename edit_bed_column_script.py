import pandas as pd

# Read the CSV file
df = pd.read_csv("realestate1.csv")

# Get unique cities
unique_cities = df['city'].unique()

# Create a mapping of cities to numbers
city_to_number = {city: i+1 for i, city in enumerate(unique_cities)}

# Replace cities with numbers in the DataFrame
df['city'] = df['city'].map(city_to_number)

# Save the modified DataFrame to the same CSV file, overwriting the original
df.to_csv("realestate1.csv", index=False)

print("City names have been replaced with numbers in the 'city' column of realestate1.csv")
