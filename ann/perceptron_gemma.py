import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.metrics import precision_score, recall_score

# Define activation function and its derivative
def sigmoid(z):
    return expit(z)

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Single-layer neural network (Perceptron)
class SingleLayerNN:
    def __init__(self, input_size, output_size, threshold=0.5):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, X, threshold=0.5):
        self.output = (sigmoid(np.dot(X, self.weights) + self.bias) >= threshold).astype(int)
        return self.output

    def backward(self, X, y, learning_rate):
        # Output layer (single layer in this case)
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Update weights and biases
        self.weights += X.T.dot(output_delta) * learning_rate
        self.bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate


def read_data():
    data = pd.read_csv("../data/breast-cancer.csv")
    return data

data = read_data()
features = data.loc[:,~data.columns.isin(['id','diagnosis'])]
predictions = data['diagnosis']
predictions_encoded = data['diagnosis'].map({'B': 0, 'M': 1})

X_train, X_test, y_train, y_test = train_test_split(features, predictions_encoded, test_size=0.2, random_state=42)

# Hyperparameters
input_size = len(features.columns)
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize the network
model = SingleLayerNN(input_size, output_size)

# Train the network
for epoch in range(epochs):
    # Forward pass
    predictions = model.forward(X_train)

    # Backward pass and weight updates
    model.backward(X_train, y_train.to_numpy().reshape(-1,1), learning_rate)

    # Calculate error (optional, for monitoring)
    error = np.mean(np.square(y_train.to_numpy().reshape(-1,1) - predictions))
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {error}")

# Test the trained model
predictions = model.forward(X_test)

# Calculate precision
precision = precision_score(y_test.to_numpy().reshape(-1,1), predictions)
print(f"Precision: {precision:.2f}")

# Calculate recall
recall = recall_score(y_test.to_numpy().reshape(-1,1), predictions)
print(f"Recall: {recall:.2f}")