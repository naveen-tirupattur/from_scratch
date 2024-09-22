import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def read_data():
    data = pd.read_csv("../data/breast-cancer.csv")
    return data

data = read_data()
features = data.loc[:,~data.columns.isin(['id','diagnosis'])]
predictions = data['diagnosis']
predictions_encoded = data['diagnosis'].map({'B': 0, 'M': 1})

X_train, X_test, y_train, y_test = train_test_split(features, predictions_encoded, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression  # Example: Logistic Regression

# 4. Create and Train the Model
model = LogisticRegression(max_iter=1000)  # Create the model instance
model.fit(X_train, y_train)  # Train the model on the training data

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# print(classification_report(y_test, y_pred, target_names=iris.target_names))
#
print(y_pred)
