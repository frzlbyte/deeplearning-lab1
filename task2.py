import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import urllib.request


# Function to load data from URLs
def load_data():
    train_url = "http://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_train.txt"
    test_url = "http://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_test.txt"

    print("Loading training data from URL...")
    train_data = np.loadtxt(urllib.request.urlopen(train_url))
    print("Loading test data from URL...")
    test_data = np.loadtxt(urllib.request.urlopen(test_url))

    # Split into features and labels
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0].astype(int)
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0].astype(int)

    # Normalize pixel values from [-1, 1] to [0, 1]
    X_train = (X_train + 1) / 2
    X_test = (X_test + 1) / 2

    # Prepare different shapes
    X_train_fcn, X_test_fcn = X_train, X_test
    X_train_local, X_test_local = X_train.reshape(-1, 16, 16, 1), X_test.reshape(-1, 16, 16, 1)

    return X_train_fcn, y_train, X_test_fcn, y_test, X_train_local, X_test_local


# Load data
X_train_fcn, y_train, X_test_fcn, y_test, X_train_local, X_test_local = load_data()

# ========== PART 1: PARAMETER INITIALIZATION ==========
print("\n===== PARAMETER INITIALIZATION =====")

# Example showing poor initialization for FCN
print("\nFCN with zeros initialization (slow learning):")
model_fcn_slow = models.Sequential([
    layers.Dense(128, activation='relu', kernel_initializer='zeros', input_shape=(256,)),
    layers.Dense(64, activation='tanh', kernel_initializer='zeros'),
    layers.Dense(32, activation='relu', kernel_initializer='zeros'),
    layers.Dense(10, activation='softmax', kernel_initializer='zeros')
])
model_fcn_slow.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_fcn_slow.fit(X_train_fcn, y_train, epochs=3, validation_data=(X_test_fcn, y_test), verbose=1)

# Example showing good initialization for FCN
print("\nFCN with he_normal initialization (effective learning):")
model_fcn_good = models.Sequential([
    layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(256,)),
    layers.Dense(64, activation='tanh', kernel_initializer='glorot_normal'),
    layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])
model_fcn_good.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_fcn_good.fit(X_train_fcn, y_train, epochs=3, validation_data=(X_test_fcn, y_test), verbose=1)

# Example showing poor initialization for CNN
print("\nCNN with ones initialization (too fast learning):")
model_cnn_fast = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='ones', input_shape=(16, 16, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='ones'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_initializer='ones'),
    layers.Dense(10, activation='softmax', kernel_initializer='ones')
])
model_cnn_fast.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn_fast.fit(X_train_local, y_train, epochs=3, validation_data=(X_test_local, y_test), verbose=1)

# ========== PART 2: LEARNING RATE ==========
print("\n===== LEARNING RATE ANALYSIS =====")

# Example showing slow learning rate for CNN
print("\nCNN with slow learning rate (0.0001):")
optimizer_slow = tf.keras.optimizers.Adam(learning_rate=0.0001)
model_cnn_lr_slow = models.Sequential([
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
model_cnn_lr_slow.compile(optimizer=optimizer_slow, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn_lr_slow.fit(X_train_local, y_train, epochs=3, validation_data=(X_test_local, y_test), verbose=1)

# Example showing effective learning rate for CNN
print("\nCNN with effective learning rate (0.001):")
optimizer_good = tf.keras.optimizers.Adam(learning_rate=0.001)
model_cnn_lr_good = models.Sequential([
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
model_cnn_lr_good.compile(optimizer=optimizer_good, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn_lr_good.fit(X_train_local, y_train, epochs=3, validation_data=(X_test_local, y_test), verbose=1)

# Example showing fast learning rate for CNN
print("\nCNN with fast learning rate (0.1):")
optimizer_fast = tf.keras.optimizers.Adam(learning_rate=0.1)
model_cnn_lr_fast = models.Sequential([
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
model_cnn_lr_fast.compile(optimizer=optimizer_fast, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn_lr_fast.fit(X_train_local, y_train, epochs=3, validation_data=(X_test_local, y_test), verbose=1)

# ========== PART 3: BATCH NORMALIZATION & BATCH SIZE ==========
print("\n===== BATCH SIZE IMPACT ON BATCH NORMALIZATION =====")

# Small batch size with batch normalization
print("\nCNN with small batch size (8):")
model_cnn_small_batch = models.Sequential([
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
model_cnn_small_batch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn_small_batch.fit(X_train_local, y_train, batch_size=8, epochs=3, validation_data=(X_test_local, y_test),
                          verbose=1)

# Large batch size with batch normalization
print("\nCNN with large batch size (128):")
model_cnn_large_batch = models.Sequential([
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
model_cnn_large_batch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn_large_batch.fit(X_train_local, y_train, batch_size=128, epochs=3, validation_data=(X_test_local, y_test),
                          verbose=1)

# ========== PART 4: MOMENTUM ==========
print("\n===== MOMENTUM ANALYSIS =====")

# CNN with different momentum values
for momentum in [0.5, 0.9, 0.99]:
    print(f"\nCNN with momentum {momentum}:")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=momentum)
    model_cnn_momentum = models.Sequential([
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
    model_cnn_momentum.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_cnn_momentum.fit(X_train_local, y_train, epochs=3, validation_data=(X_test_local, y_test), verbose=1)

# ========== BASELINE MODELS ==========
print("\n===== BASELINE MODELS =====")

# Define and train baseline models
model_fcn = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(256,)),
    layers.Dense(64, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_fcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_local = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=(16, 16, 1)),
    layers.Conv2D(64, (3, 3), activation='tanh', padding='valid'),
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

# Train models with reduced epochs for faster testing
print("\nTraining baseline models with default settings...")
history_fcn = model_fcn.fit(X_train_fcn, y_train, epochs=3, validation_data=(X_test_fcn, y_test), verbose=1)
history_local = model_local.fit(X_train_local, y_train, epochs=3, validation_data=(X_test_local, y_test), verbose=1)
history_cnn = model_cnn.fit(X_train_local, y_train, epochs=3, validation_data=(X_test_local, y_test), verbose=1)

# Evaluate models
fcn_eval = model_fcn.evaluate(X_test_fcn, y_test, verbose=0)
local_eval = model_local.evaluate(X_test_local, y_test, verbose=0)
cnn_eval = model_cnn.evaluate(X_test_local, y_test, verbose=0)

# Display baseline results
results_df = pd.DataFrame({
    "Model": ["Fully Connected", "Locally Connected", "Convolutional Neural Network"],
    "Loss": [fcn_eval[0], local_eval[0], cnn_eval[0]],
    "Accuracy": [fcn_eval[1], local_eval[1], cnn_eval[1]]
})

print("\nBaseline Model Results:")
print(results_df)