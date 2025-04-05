# 1. Import all the required Python Libraries
import pandas as pd          # For data handling and analysis
import numpy as np           # For numerical operations
import seaborn as sns        # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs

# 2. Load the dataset into pandas dataframe
# Assume the dataset file is named "StudentsPerformance.csv" and is present in the same folder as your VS Code project
df = pd.read_csv("exams.csv")

# 3. Initial view of dataset
print("🔹 First 5 rows of the dataset:")
print(df.head())

# 4. Data Preprocessing
print("\n🔹 Check for missing values:")
print(df.isnull().sum())  # Checks if there are any missing values

print("\n🔹 Summary statistics of the dataset:")
print(df.describe())  # Summary stats of numeric columns

print("\n🔹 Shape of the dataset (rows, columns):")
print(df.shape)  # Number of rows and columns

print("\n🔹 Column names in the dataset:")
print(df.columns.tolist())

# 5. Variable Descriptions and Types
print("\n🔹 Data types of variables:")
print(df.dtypes)

# Rename columns for easier access (optional)
df.columns = ['gender', 'race', 'parent_education', 'lunch', 'test_prep', 'math_score', 'reading_score', 'writing_score']

# If any data types are incorrect, convert them properly
# For example, ensure numerical scores are in integer type
df['math_score'] = df['math_score'].astype(int)
df['reading_score'] = df['reading_score'].astype(int)
df['writing_score'] = df['writing_score'].astype(int)

# 6. Convert Categorical variables into Quantitative variables
# We will use Label Encoding and One-Hot Encoding

from sklearn.preprocessing import LabelEncoder

# Label Encoding for gender and lunch (binary)
label_enc = LabelEncoder()
df['gender_encoded'] = label_enc.fit_transform(df['gender'])  # male=1, female=0
df['lunch_encoded'] = label_enc.fit_transform(df['lunch'])    # standard=1, free/reduced=0

# One-Hot Encoding for race/ethnicity, parental education, and test preparation course
df = pd.get_dummies(df, columns=['race', 'parent_education', 'test_prep'], drop_first=True)

print("\n🔹 Dataset after encoding:")
print(df.head())

# 7. Final Check
print("\n🔹 Data types after encoding:")
