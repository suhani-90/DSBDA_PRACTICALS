import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
titanic = pd.read_csv('Titanic.csv')

# Optional: Check for missing values in 'Age' and drop them for the box plot
titanic = titanic.dropna(subset=['Age'])

# Convert 'Survived' to a readable format (optional, for better labels)
titanic['Survived'] = titanic['Survived'].map({0: 'No', 1: 'Yes'})

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=titanic, x='Sex', y='Age', hue='Survived', palette='Set2')
plt.title('Age Distribution by Gender and Survival Status')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.grid(True)
plt.legend(title='Survived')
plt.show()
