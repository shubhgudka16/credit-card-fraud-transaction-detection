import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
credit_card_data = pd.read_csv("creditcard.csv")

# Display first 5 rows
print(credit_card_data.head())

# Dataset information
print(credit_card_data.info())

# Check missing values
print(credit_card_data.isnull().sum())

# Count legitimate and fraud transactions
print(credit_card_data['Class'].value_counts())

# Separate legitimate and fraud transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# Statistical summary
print(legit['Amount'].describe())
print(fraud['Amount'].describe())

# Compare mean values
print(credit_card_data.groupby('Class').mean())

# Create balanced dataset
legit_sample = legit.sample(n=492, random_state=2)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

print(new_dataset.head())
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

# Split features and target
X = new_dataset.drop(columns=['Class'])
Y = new_dataset['Class']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y,random_state=2)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, Y_train)

# Training accuracy
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)

print("Accuracy on Training Data:", training_accuracy)

# Testing accuracy
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Accuracy on Test Data:", test_accuracy)

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(Y_test, X_test_prediction)
print(cm)

# Classification Report
print("Classification Report:")
print(classification_report(Y_test, X_test_prediction))

row = int(input("Enter row number from the dataset: "))

sample = credit_card_data.drop(columns=['Class']).iloc[row].values.reshape(1, -1)
actual = credit_card_data['Class'].iloc[row]

prediction = model.predict(sample)

print("Actual Class:", actual)

if prediction[0] == 0:
    print("Predicted: Legitimate Transaction")
else:
    print("Predicted: Fraudulent Transaction")