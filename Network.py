import numpy as np
from Layer import Layer
import pickle
class Network:
    def __init__(self, number_of_nodes):
        self.layers = []
        self.number_of_nodes = number_of_nodes
        for i in range(len(number_of_nodes) - 1):
            self.layers.append(Layer(number_of_nodes[i], number_of_nodes[i + 1]))

    def forward(self, inputs):
        forward_output = inputs
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                forward_output = self.layers[i].forward(forward_output, "softmax")
            else:
                forward_output = self.layers[i].forward(forward_output, "relu")
        return forward_output

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # Avoid log(0) error
        return loss

    def backward(self, y_true, y_pred):
        dA = y_pred - y_true
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                dA = self.layers[i].backward(dA, "softmax")
            else:
                dA = self.layers[i].backward(dA, "relu")

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def compute_accuracy(self, y_true, y_pred):
        # Get the predicted class (index of max probability for softmax output)
        predicted_classes = np.argmax(y_pred, axis=1)
        
        # Get the true classes (index of 1 in one-hot encoded vectors)
        true_classes = np.argmax(y_true, axis=1)
        
        # Compare predictions to true labels
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def train(self, X, y, epochs, learning_rate, batch_size=64):
        m = X.shape[0]  # Number of samples
        for epoch in range(epochs):
            # Shuffle the data for mini-batch training
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0

            # Process in batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                self.backward(y_batch, y_pred)
                self.update_parameters(learning_rate)
                accuracy = self.compute_accuracy(y_batch, y_pred)

                epoch_loss += loss
                epoch_accuracy += accuracy
                num_batches += 1

            # Average loss and accuracy over all batches
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches

            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy * 100:.2f}%')

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            for layer in self.layers:
                pickle.dump((layer.weights, layer.bias), f)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            for layer in self.layers:
                layer.weights, layer.bias = pickle.load(f)
