import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Iris_dataset/Iris.csv')

# Drop 'Id' column if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Plotting histograms for numeric features
numeric_features = df.select_dtypes(include='number').columns

plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_features):
    plt.subplot(2, 2, i + 1)
    plt.hist(df[col], bins=15, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
