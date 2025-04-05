import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load dataset
df = pd.read_csv('Iris_dataset/Iris.csv')

# Drop 'Id' column
df = df.drop(columns=['Id'])

# Features and target
X = df.drop(columns=['Species'])  # SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
y = df['Species']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=y.unique())
print("Confusion Matrix:")
print(conf_matrix)

# Metrics Calculation
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro')  # macro = equally weighted
recall = recall_score(y_test, y_pred, average='macro')

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Breakdown of TP, FP, TN, FN for each class
print("\nTrue Positives, False Positives, True Negatives, False Negatives per class:")

labels = y.unique()
for i, label in enumerate(labels):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)
    print(f"\nClass: {label}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
