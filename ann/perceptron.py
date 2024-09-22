import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from scipy.special import expit

for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def sigmoid(x):
    return expit(x)

def perceptron(x, w, b):
    return sigmoid(np.dot(x, w) + b)

def init_weights(n):
    w = np.random.randn(n)
    b = np.random.randn()
    return w, b


def train_perceptron(x, y, w, b, epochs=10000, lr=0.1, threshold=0.5):
    y_predictions = pd.DataFrame()
    for epoch in range(epochs):
        print(epoch)
        for i in range(len(x)):
            row = x.iloc[i,:]
            y_pred = perceptron(row, w, b)
            y_pred = (y_pred >= threshold).astype(int)
            y_predictions.loc[i,'predictions'] = y_pred
            w += lr * (y.iloc[i] - y_pred) * y_pred * (1 - y_pred) * row
            b += lr * (y.iloc[i] - y_pred) * y_pred * (1 - y_pred)
        mse = np.mean(np.square(y - y_predictions['predictions']))
        print(f"MSE: {mse:.4f}")
    return w, b

def read_data():
    data = pd.read_csv("../data/breast-cancer.csv")
    return data

data = read_data()
features = data.loc[:,~data.columns.isin(['id','diagnosis'])]
predictions = data['diagnosis']
predictions_encoded = data['diagnosis'].map({'B': 0, 'M': 1})
X_train, X_test, y_train, y_test = train_test_split(features, predictions_encoded, test_size=0.2, random_state=42)

w,b = init_weights(len(features.columns))
w,b = train_perceptron(X_train,y_train,w,b)

print(w)
print(b)
