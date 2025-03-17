# -A1_CE24D903_DA6401
# Feed forward Neural Network Training 

## Overview
This script implements a customizable neural network using NumPy and supports various configurations, including user-defined architecture, initialization methods, activation functions, and optimizers (SGD, Momentum, NAG, RMSProp, Adam).

## Features

- **Datasets**: MNIST, Fashion MNIST
- **Customizable Neural Network Architecture**
- **Weight Initialization**: Random, Xavier
- **Activation Functions**: Sigmoid, ReLU, Tanh, Identity
- Supports forward propagation and softmax classification
- Loss functions: Mean Squared Error (MSE), Cross-Entropy
- Optimizers: Stochastic Gradient Descent (SGD), Momentum, Nesterov Accelerated Gradient (NAG), RMSProp, Adam

## How to Use

### 1. Load Dataset
```python
x_train, y_train, x_test, y_test = load_dataset("fashion_mnist")
```

### 2. Define Network Architecture
```python
layer_sizes = get_user_defined_layers(num_hidden_layers=3, neurons_per_layer=[4, 5, 6])
```

### 3. Initialize Weights and Biases
```python
weights, biases = initialize_weights_and_biases(layer_sizes, init_method="random")
```

### 4. Forward Propagation
```python
h_values, a_values = forward_propagation(x_train, weights, biases, activation="relu")
```

### 5. Compute Accuracy
```python
accuracy = compute_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}%")
```

### 6. Train Model with Optimizers

#### Stochastic Gradient Descent (SGD)
```python
test_accuracy = sgd(x_train, y_train, weights, biases, learning_rate=0.01, num_epochs=10)
print(f"SGD Test Accuracy: {test_accuracy:.2f}%")
```

#### Momentum Optimizer
```python
weights, biases = momentum(x_train, y_train, weights, biases, learning_rate=0.01, momentum_coef=0.9, num_epochs=10)
y_pred_momentum = predict(x_test, weights, biases)
accuracy_momentum = compute_accuracy(y_test, y_pred_momentum)
print(f"Momentum Test Accuracy: {accuracy_momentum:.2f}%")
```

#### Nesterov Accelerated Gradient (NAG)
```python
test_accuracy_nag = nag(x_train, y_train, weights, biases, learning_rate=0.01, momentum_coef=0.9, num_epochs=10)
print(f"NAG Test Accuracy: {test_accuracy_nag:.2f}%")
```

#### RMSProp
```python
test_accuracy_rmsprop = rmsprop(x_train, y_train, weights, biases, learning_rate=0.001, beta=0.9, eps=1e-8, num_epochs=10)
print(f"RMSProp Test Accuracy: {test_accuracy_rmsprop:.2f}%")
```

#### Adam Optimizer
```python
test_accuracy_adam = adam(x_train, y_train, weights, biases, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, num_epochs=10)
print(f"Adam Test Accuracy: {test_accuracy_adam:.2f}%")
```

## Evaluation
```python
y_pred = predict(x_test, weights, biases)
accuracy = compute_accuracy(y_test, y_pred)
print(f"Final Test Accuracy: {accuracy:.2f}%")
```

## Usage
Modify parameters such as `num_hidden_layers`, `neurons_per_layer`, `learning_rate`, and `num_epochs` in the script to tailor the model to your needs.

## Dependencies
- NumPy
- Keras (for dataset loading only)

Install dependencies using:
```bash
pip install numpy keras
```
