# 1. Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 2. Load dataset
# Make sure the file is in the same directory as this Python file
df = pd.read_csv("exams.csv")

# Show the first few rows
print("🔹 First 5 rows:")
print(df.head())

# ----------------------------------------------
# 1. MISSING VALUES & INCONSISTENCIES
# ----------------------------------------------

# Check for missing values
print("\n🔹 Missing values in each column:")
print(df.isnull().sum())

# Check for inconsistencies (e.g., unexpected strings)
print("\n🔹 Unique values in each categorical column:")
for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    print(f"{col}: {df[col].unique()}")

# Example: Fill missing numeric values with mean (if any)
numeric_cols = ['math score', 'reading score', 'writing score']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Example: Fill missing categorical with mode
cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ----------------------------------------------
# 2. OUTLIER DETECTION & HANDLING
# ----------------------------------------------

# Using IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Show outlier rows
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    print(f"\n🔹 Outliers detected in '{column}': {len(outliers)}")
    # Remove outliers
    return data[(data[column] >= lower) & (data[column] <= upper)]

# Apply outlier removal
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

# ----------------------------------------------
# 3. DATA TRANSFORMATION
# ----------------------------------------------

# Let's apply log transformation to 'math score' to reduce skewness
# Reason: Exam scores can be slightly skewed. Log transforms compress large values.

# Before transformation
plt.figure(figsize=(8,4))
sns.histplot(df['math score'], kde=True, color='orange')
plt.title("Before Log Transformation - Math Score")
plt.show()

# Check if all values are > 0 before log
if (df['math score'] <= 0).any():
    # Add a small constant if needed
    df['math_score_log'] = np.log1p(df['math score'])
else:
    df['math_score_log'] = np.log(df['math score'])

# After transformation
plt.figure(figsize=(8,4))
sns.histplot(df['math_score_log'], kde=True, color='green')
plt.title("After Log Transformation - Math Score")
plt.show()

# ----------------------------------------------
# Final Dataset Check
# ----------------------------------------------
print("\n🔹 Final dataset shape:", df.shape)
print("\n🔹 Final columns:", df.columns.tolist())
print(df.head())

