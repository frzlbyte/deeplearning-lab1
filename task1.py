import numpy as np
import urllib.request
from tensorflow.keras import layers, models

# Load the dataset from the provided URLs
train_url = "http://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_train.txt"
test_url = "http://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_test.txt"

# Download and read the data
print("Loading training data...")
train_data = np.loadtxt(urllib.request.urlopen(train_url))
print("Loading test data...")
test_data = np.loadtxt(urllib.request.urlopen(test_url))

# Split into features and labels
X_train = train_data[:, 1:]  # Features (256 features per image)
y_train = train_data[:, 0].astype(int)  # Labels (digits 0-9)
X_test = test_data[:, 1:]  # Features
y_test = test_data[:, 0].astype(int)  # Labels

# Normalize the features to be between 0 and 1
X_train = (X_train + 1) / 2  # Rescale to [0, 1]
X_test = (X_test + 1) / 2

# Prepare data for different model architectures

# Flat data for fully connected network
X_train_fcn = X_train
X_test_fcn = X_test

# 1D data for locally connected network
X_train_local = X_train.reshape(-1, 256, 1)
X_test_local = X_test.reshape(-1, 256, 1)

# 2D data for CNN
X_train_cnn = X_train.reshape(-1, 16, 16, 1)
X_test_cnn = X_test.reshape(-1, 16, 16, 1)

print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")

# ======= Model 1: Fully Connected Network =======
print("\nCreating and training fully connected network...")
model_fcn = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(256,)),  # First dense layer with ReLU activation
    layers.Dense(64, activation='tanh'),  # Second dense layer with Tanh activation
    layers.Dense(32, activation='relu'),  # Third dense layer with ReLU activation
    layers.Dense(10, activation='softmax')  # Output layer with softmax activation for classification
])

# Compile the model
model_fcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the digit dataset
history_fcn = model_fcn.fit(X_train_fcn, y_train, epochs=10, batch_size=32, validation_data=(X_test_fcn, y_test))

# Model summary
model_fcn.summary()

# ======= Model 2: Locally Connected Network (No Weight Sharing) =======
print("\nCreating and training locally connected network (no weight sharing)...")
model_locally_connected = models.Sequential([
    layers.Conv1D(128, kernel_size=5, activation='relu', input_shape=(256, 1)),  # Using Conv1D for local connectivity
    layers.Conv1D(64, kernel_size=5, activation='tanh'),  # Using tanh activation in one layer
    layers.Conv1D(32, kernel_size=5, activation='relu'),  # Using ReLU activation in another layer
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model_locally_connected.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_local = model_locally_connected.fit(X_train_local, y_train, epochs=10, batch_size=32, validation_data=(X_test_local, y_test))

# Model summary
model_locally_connected.summary()

# ======= Model 3: Convolutional Neural Network (Weight Sharing) =======
print("\nCreating and training convolutional neural network (weight sharing)...")
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)),  # First conv layer with ReLU
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='tanh'),  # Second conv layer with tanh
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (2, 2), activation='relu'),  # Third conv layer with ReLU
    layers.Flatten(),
    layers.Dense(10, activation='softmax')  # Output layer
])

# Compile and train the model
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cnn = model_cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))

# Model summary
model_cnn.summary()

# ======= Evaluate all models =======
print("\nEvaluating models on test data...")
fcn_loss, fcn_acc = model_fcn.evaluate(X_test_fcn, y_test, verbose=0)
local_loss, local_acc = model_locally_connected.evaluate(X_test_local, y_test, verbose=0)
cnn_loss, cnn_acc = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)

print(f"\nModel Comparison:")
print(f"Fully Connected Network     - Test Accuracy: {fcn_acc:.4f}, Loss: {fcn_loss:.4f}")
print(f"Locally Connected Network   - Test Accuracy: {local_acc:.4f}, Loss: {local_loss:.4f}")
print(f"Convolutional Neural Network - Test Accuracy: {cnn_acc:.4f}, Loss: {cnn_loss:.4f}")