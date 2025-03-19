Task 1: Fully Connected Neural Network Example:

Loading the Digital Datasets:

import numpy as np
import urllib.request

# Load the dataset from the provided URLs
train_url = "http://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_train.txt"
test_url = "http://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_test.txt"

# Download and read the data
train_data = np.loadtxt(urllib.request.urlopen(train_url))
test_data = np.loadtxt(urllib.request.urlopen(test_url))

# Split into features and labels
X_train = train_data[:, 1:]  # Features (256 features per image)
y_train = train_data[:, 0].astype(int)  # Labels (digits 0-9)

X_test = test_data[:, 1:]  # Features
y_test = test_data[:, 0].astype(int)  # Labels

# Normalize the features to be between 0 and 1
X_train = (X_train + 1) / 2  # Rescale to [0, 1]
X_test = (X_test + 1) / 2

# Reshape data to fit the model input (add a channel dimension for CNNs)
X_train = X_train.reshape(-1, 256, 1)  # For the locally connected networks (1D conv)
X_test = X_test.reshape(-1, 256, 1)  # For the locally connected networks (1D conv)

# For CNN (2D conv), reshape the data to 16x16 images (e.g., for 2D convolutions)
X_train_cnn = X_train.reshape(-1, 16, 16, 1)  # Reshape to 16x16 images
X_test_cnn = X_test.reshape(-1, 16, 16, 1)  # Reshape to 16x16 images



-------------------------------------------------------------------------------------------------------------------------------------------------



# Import necessary libraries
from tensorflow.keras import layers, models

# Define the first fully connected network
model_fcn = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(256,)),  # First dense layer with ReLU activation
    layers.Dense(64, activation='tanh'),  # Second dense layer with Tanh activation
    layers.Dense(32, activation='relu'),  # Third dense layer with ReLU activation
    layers.Dense(10, activation='softmax')  # Output layer with softmax activation for classification
])

# Compile the model
model_fcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the digit dataset
model_fcn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Model summary
model_fcn.summary()

Model 2 Locally Connected Network Example:

model_locally_connected_no_weights = models.Sequential([
    layers.Conv1D(128, kernel_size=5, activation='relu', input_shape=(256, 1)),
    layers.Conv1D(64, kernel_size=5, activation='tanh'),
    layers.Conv1D(32, kernel_size=5, activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model_locally_connected_no_weights.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_locally_connected_no_weights.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Model summary
model_locally_connected_no_weights.summary()


Convolutional Neural Network Example:

model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='tanh'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (2, 2), activation='relu'),  # Use kernel size (2x2) for the last Conv2D layer
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))

# Model summary
model_cnn.summary()

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_cnn.summary()
