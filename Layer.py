import numpy as np
import pickle

def generate_matrix_of_weights(num_inputs, num_outputs):
    # Xavier initialization
    return np.random.randn(num_inputs, num_outputs) * np.sqrt(2 / (num_inputs + num_outputs))

class Layer:
    def __init__(self, num_inputs: int, num_outputs: int):
        if num_outputs <= 0:
            raise ValueError("Number of outputs must be greater than zero.")
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = generate_matrix_of_weights(num_inputs, num_outputs)
        self.bias = np.zeros(num_outputs)

    def forward(self, inputs, activation_function):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        if activation_function == "sigmoid":
            self.activated_output = self.sigmoid(self.output)
        elif activation_function == "relu":
            self.activated_output = self.relu(self.output)
        elif activation_function == "softmax":
            self.activated_output = self.softmax(self.output)
        else:
            self.activated_output = self.output
        return self.activated_output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def relu_deriv(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        # Subtract max to prevent overflow
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, error_vs_activation, activation_function):
        if activation_function == "sigmoid":
            error_vs_dot_product = error_vs_activation * self.sigmoid_deriv(self.activated_output)
        elif activation_function == "relu":
            error_vs_dot_product = error_vs_activation * self.relu_deriv(self.activated_output)
        else:
            error_vs_dot_product = error_vs_activation

        error_vs_weights = np.dot(self.inputs.T, error_vs_dot_product)
        error_vs_biases = np.sum(error_vs_dot_product, axis=0, keepdims=True)
        error_vs_activation_previous = np.dot(error_vs_dot_product, self.weights.T)
        
        self.error_vs_weights = error_vs_weights
        self.error_vs_biases = error_vs_biases.flatten()

        return error_vs_activation_previous

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.error_vs_weights
        self.bias -= learning_rate * self.error_vs_biases

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.bias), f)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            self.weights, self.bias = pickle.load(f)
