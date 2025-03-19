import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd

# Function to load data
def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            labels.append(int(values[0]))  # First column is the label
            data.append([float(v) for v in values[1:]])  # Remaining 256 values are pixel features
    return np.array(data), np.array(labels)

# Load datasets
X_train, y_train = load_data('zip_train.txt')
X_test, y_test = load_data('zip_test.txt')

# Normalize data to range [0, 1]
X_train = (X_train + 1) / 2
X_test = (X_test + 1) / 2

# Reshape data
X_train_fcn, X_test_fcn = X_train, X_test  # For Fully Connected Network
X_train_conv, X_test_conv = X_train.reshape(-1, 16, 16, 1), X_test.reshape(-1, 16, 16, 1)  # For CNN & Locally Connected

# Define and compile models
model_fcn = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(256,)),
    layers.Dense(64, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_fcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_local = models.Sequential([
    layers.LocallyConnected2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)),
    layers.LocallyConnected2D(64, (3, 3), activation='tanh'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_local.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train models
history_fcn = model_fcn.fit(X_train_fcn, y_train, epochs=10, validation_data=(X_test_fcn, y_test), verbose=1)
history_local = model_local.fit(X_train_conv, y_train, epochs=10, validation_data=(X_test_conv, y_test), verbose=1)
history_cnn = model_cnn.fit(X_train_conv, y_train, epochs=10, validation_data=(X_test_conv, y_test), verbose=1)

# Evaluate models
fcn_eval = model_fcn.evaluate(X_test_fcn, y_test, verbose=0)
local_eval = model_local.evaluate(X_test_conv, y_test, verbose=0)
cnn_eval = model_cnn.evaluate(X_test_conv, y_test, verbose=0)

# Save results
results_df = pd.DataFrame({
    "Model": ["Fully Connected", "Locally Connected", "Convolutional Neural Network"],
    "Loss": [fcn_eval[0], local_eval[0], cnn_eval[0]],
    "Accuracy": [fcn_eval[1], local_eval[1], cnn_eval[1]]
})

print(results_df)
results_df.to_csv("model_performance.csv", index=False)
