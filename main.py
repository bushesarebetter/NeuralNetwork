import numpy as np
import pandas as pd
from PIL import Image
from Layer import Layer
from Network import Network

# Function to load and preprocess the image
def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels
        img_array = np.array(img)  # Convert to NumPy array
        img_array = img_array.flatten()  # Flatten to 1D array of size 784
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        return img_array

# Main Code for Training

training = False


# Initialize the network
number_of_nodes = [784, 128, 10] # output layer neurons should match up to num_classes
network = Network(number_of_nodes)

if training:
    # One-hot encode the answers
    num_classes = 10
    imported_data = pd.read_csv('digit-recognizer\\mnist_train.csv', header=None)
    answers = imported_data.iloc[:, 0].values  # Convert to NumPy array
    training_data = imported_data.iloc[:, 1:].values / 255.0  # Convert to NumPy array
    answers_one_hot = np.eye(num_classes)[answers]
    network.train(training_data, answers_one_hot, epochs=100, learning_rate=0.01, batch_size=64)

    # Save the weights
    network.save_weights('network_weights.pkl')
else:
    # Load the weights
    network.load_weights('network_weights.pkl')

# Load and preprocess the custom image
image_path = 'testing_images\\NINE.png'  # Replace with the path to your image
custom_image = load_image(image_path)

# Make a prediction
custom_image = custom_image.reshape(1, -1)  # Reshape to match the input shape expected by the network
prediction = network.forward(custom_image)
predicted_class = np.argmax(prediction, axis=1)

print(f"Prediction: {prediction}")
print("Top 3 Predictions:")
top_3_indices = np.argsort(prediction[0])[-3:][::-1]
for i, idx in enumerate(top_3_indices):
    print(f"{i+1}. Class {idx} with probability {prediction[0][idx]}")
print(f'Predicted class: {predicted_class[0]}')
