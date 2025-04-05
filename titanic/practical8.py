import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV file (change the path if needed)
titanic = pd.read_csv('Titanic.csv')  # Replace with the correct file path

# Display the first few rows to verify the data
print(titanic.head())

# Plot a histogram for the 'Fare' column
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='Fare', bins=30, kde=True, color='teal')
plt.title('Distribution of Ticket Fare')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.show()
