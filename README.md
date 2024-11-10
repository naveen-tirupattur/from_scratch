Let's break down the math behind training a perceptron with backpropagation.

**Understanding the Perceptron**

A perceptron is the simplest kind of neural network. It takes several inputs (x₁, x₂, ... xn), multiplies each input by a weight (w₁, w₂, ... wn), sums them up, adds a bias (b), and then applies an activation function to the result.

Here's the equation:

```
y = f(∑(wi * xi) + b) 
```

*  **y** = output
*  **f()** = activation function (e.g., sigmoid function)
*  **wi** = weight associated with input xi
*  **xi** = input value
*  **b** = bias

**Backpropagation: Adjusting Weights and Bias**

Backpropagation is the algorithm that adjusts the weights (w) and bias (b) to minimize the error between the perceptron's output (y) and the desired output (target, often represented as 't').

Here's a step-by-step mathematical explanation:

1. **Calculate the Error (Cost Function):**

   We need a way to measure how "off" our perceptron is. A common choice is the mean squared error:

   ```
   Error (E) = 1/2 * (t - y)² 
   ```

    *  **t** = target (desired output)
    *  **y** = actual output

2. **Update the Weights and Bias:**

   We want to find the direction and amount to adjust the weights and bias to reduce the error. We use the gradient descent algorithm, which involves finding the partial derivatives of the error function with respect to each weight and the bias.

    * **Weight Update:**
      ```
      wi = wi - learning_rate * ∂E/∂wi
      ```
    * **Bias Update:**
      ```
      b = b - learning_rate * ∂E/∂b
      ```
    * **learning_rate** is a small positive number (hyperparameter) that controls how big of a step we take in the direction of the gradient.

3. **Calculating the Gradients:**

   To update the weights and bias, we need to calculate those partial derivatives (∂E/∂wi and ∂E/∂b).  This is where the chain rule of calculus comes in:

    * **Output Gradient:**
       ```
       ∂E/∂y = (y - t) 
       ```
    * **Activation Function Gradient (example: sigmoid):**
       ```
       ∂y/∂z = y * (1 - y)   // Assuming f(z) = 1 / (1 + exp(-z)) is the sigmoid function
       ```
      where z = ∑(wi * xi) + b
    * **Weight Gradient:**
       ```
       ∂E/∂wi = ∂E/∂y * ∂y/∂z * ∂z/∂wi =  (y - t) * y * (1 - y) * xi
       ```
    * **Bias Gradient:**
       ```
       ∂E/∂b = ∂E/∂y * ∂y/∂z * ∂z/∂b = (y - t) * y * (1 - y) 
       ```

4. **Iterate:**

   Repeat steps 1-3 with many training examples. Each iteration through the entire training dataset is called an epoch. The perceptron gradually learns the optimal weights and bias to make accurate predictions.

**Simplified Example (Single Input):**

Let's say we have a single input perceptron:

* Input (x) = 1
* Target (t) = 0
* Initial weight (w) = 0.5
* Initial bias (b) = 0.2
* Learning rate = 0.1

1. **Calculate Output:**
   ```
   z = (0.5 * 1) + 0.2 = 0.7
   y = sigmoid(0.7) ≈ 0.67  // Using the sigmoid activation function
   ```

2. **Calculate Error:**
   ```
   Error = 1/2 * (0 - 0.67)² ≈ 0.22
   ```

3. **Calculate Gradients:**
   ```
   ∂E/∂y = (0.67 - 0) = 0.67
   ∂y/∂z = 0.67 * (1 - 0.67) ≈ 0.22 
   ∂E/∂w = 0.67 * 0.22 * 1 ≈ 0.15
   ∂E/∂b = 0.67 * 0.22 ≈ 0.15
   ```

4. **Update Weight and Bias:**
   ```
   w = 0.5 - (0.1 * 0.15) = 0.485
   b = 0.2 - (0.1 * 0.15) = 0.185 
   ```

You would repeat this process with many more training examples to train the perceptron.

**Important Notes:**

* Backpropagation is more commonly used in multi-layer neural networks rather than simple perceptrons.
* The choice of activation function affects the gradient calculations.
* Hyperparameters like the learning rate need to be tuned for optimal performance.
* Imagine you have a very large input: 
  * Direct Calculation: y_pred * (1 - y_pred) might result in very small values, leading to very small updates, potentially slowing down learning.
  * sigmoid_derivative: The sigmoid_derivative function is designed to handle these extreme values more accurately.

This breakdown provides a mathematical foundation for understanding how backpropagation works to train a perceptron. 
