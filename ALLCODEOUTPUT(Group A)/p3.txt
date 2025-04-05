import pandas as pd

# Load the dataset
df = pd.read_csv('Iris_dataset/Iris.csv')

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Drop the 'Id' column as it's not useful for analysis
df = df.drop(columns=['Id'])

# Task 1: Summary statistics grouped by the categorical variable 'Species'
grouped_stats = df.groupby('Species').agg(['mean', 'median', 'min', 'max', 'std'])

print("\nTask 1: Summary statistics (mean, median, min, max, std) grouped by 'Species':")
print(grouped_stats)

# Task 1 (continued): Create a list that contains numeric values for each species
# Example: SepalLengthCm values grouped by species
sepal_length_lists = df.groupby('Species')['SepalLengthCm'].apply(list).to_dict()

print("\nTask 1 (continued): List of SepalLengthCm values for each Species:")
for species, values in sepal_length_lists.items():
    print(f"{species}: {values[:5]} ...")  # Displaying only first 5 values for brevity

# Task 2: Basic statistical details (describe) for each species
species_list = df['Species'].unique()

print("\nTask 2: Basic statistical details for each Species:")
for species in species_list:
    print(f"\nStatistics for {species}:")
    stats = df[df['Species'] == species].describe()
    print(stats)
