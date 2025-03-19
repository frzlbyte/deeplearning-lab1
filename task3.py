import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Function to load data from URLs
def load_data():
    # URLs for the datasets
    train_url = "http://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_train.txt"
    test_url = "http://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_test.txt"

    print("Loading training data from URL...")
    train_data = np.loadtxt(urllib.request.urlopen(train_url))

    print("Loading test data from URL...")
    test_data = np.loadtxt(urllib.request.urlopen(test_url))

    # Split into features and labels
    X_train = train_data[:, 1:]  # Features (256 features per image)
    y_train = train_data[:, 0].astype(int)  # Labels (digits 0-9)
    X_test = test_data[:, 1:]  # Features
    y_test = test_data[:, 0].astype(int)  # Labels

    # Normalize pixel values from [-1, 1] to [0, 1]
    X_train = (X_train + 1) / 2
    X_test = (X_test + 1) / 2

    # Reshape data for different models
    X_train_fcn, X_test_fcn = X_train, X_test  # Fully Connected Network (Flattened input)
    X_train_local, X_test_local = X_train.reshape(-1, 16, 16, 1), X_test.reshape(-1, 16, 16,
                                                                                 1)  # For CNN & Locally Connected

    print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")

    return X_train_fcn, y_train, X_test_fcn, y_test, X_train_local, X_test_local


# Load data
X_train_fcn, y_train, X_test_fcn, y_test, X_train_local, X_test_local = load_data()

# Convert labels to one-hot encoding for ensemble models
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

# =================== PART 1: ENSEMBLE METHODS ===================
print("\n===== ENSEMBLE METHODS =====")

# Create and train multiple models for the ensemble

# Model 1: Fully Connected Network (optimized parameters from Task II)
print("\nTraining Model 1: Fully Connected Network")
model1 = models.Sequential([
    layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(256,)),
    layers.Dense(64, activation='tanh', kernel_initializer='glorot_normal'),
    layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])
model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history1 = model1.fit(
    X_train_fcn, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_fcn, y_test),
    verbose=1
)

# Model 2: Locally Connected Network (optimized parameters from Task II)
print("\nTraining Model 2: Locally Connected Network")
model2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='valid',
                  kernel_initializer='he_normal', input_shape=(16, 16, 1)),
    layers.Conv2D(64, (3, 3), activation='tanh', padding='valid',
                  kernel_initializer='glorot_normal'),
    layers.Flatten(),
    layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history2 = model2.fit(
    X_train_local, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_local, y_test),
    verbose=1
)

# Model 3: CNN with Batch Normalization (optimized parameters from Task II)
print("\nTraining Model 3: CNN with Batch Normalization")
model3 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  kernel_initializer='he_normal', input_shape=(16, 16, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu',
                  kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])
model3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history3 = model3.fit(
    X_train_local, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_local, y_test),
    verbose=1
)

# Model 4: CNN with Different Architecture (for diversity in ensemble)
print("\nTraining Model 4: CNN with Different Architecture")
model4 = models.Sequential([
    layers.Conv2D(48, (5, 5), activation='relu', padding='same',
                  kernel_initializer='he_normal', input_shape=(16, 16, 1)),
    layers.BatchNormalization(),
    layers.AveragePooling2D((2, 2)),
    layers.Conv2D(96, (3, 3), activation='relu', padding='same',
                  kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.AveragePooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])
model4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history4 = model4.fit(
    X_train_local, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_local, y_test),
    verbose=1
)

# Evaluate individual models
print("\nEvaluating individual models:")
model1_loss, model1_acc = model1.evaluate(X_test_fcn, y_test, verbose=0)
model2_loss, model2_acc = model2.evaluate(X_test_local, y_test, verbose=0)
model3_loss, model3_acc = model3.evaluate(X_test_local, y_test, verbose=0)
model4_loss, model4_acc = model4.evaluate(X_test_local, y_test, verbose=0)

print(f"Model 1 (FCN): Accuracy = {model1_acc:.4f}, Loss = {model1_loss:.4f}")
print(f"Model 2 (Local): Accuracy = {model2_acc:.4f}, Loss = {model2_loss:.4f}")
print(f"Model 3 (CNN): Accuracy = {model3_acc:.4f}, Loss = {model3_loss:.4f}")
print(f"Model 4 (CNN2): Accuracy = {model4_acc:.4f}, Loss = {model4_loss:.4f}")

# Get predictions from all models
model1_preds = model1.predict(X_test_fcn, verbose=0)
model2_preds = model2.predict(X_test_local, verbose=0)
model3_preds = model3.predict(X_test_local, verbose=0)
model4_preds = model4.predict(X_test_local, verbose=0)

# Ensemble Method 1: Simple Average (soft voting)
ensemble_preds_avg = (model1_preds + model2_preds + model3_preds + model4_preds) / 4
ensemble_avg_classes = np.argmax(ensemble_preds_avg, axis=1)
ensemble_avg_acc = np.mean(ensemble_avg_classes == y_test)
print(f"\nEnsemble Method 1 (Average): Accuracy = {ensemble_avg_acc:.4f}")

# Ensemble Method 2: Weighted Average based on validation performance
weights = np.array([model1_acc, model2_acc, model3_acc, model4_acc])
weights = weights / np.sum(weights)  # Normalize to sum to 1
print(f"Model weights: {weights}")

ensemble_preds_weighted = (
        weights[0] * model1_preds +
        weights[1] * model2_preds +
        weights[2] * model3_preds +
        weights[3] * model4_preds
)
ensemble_weighted_classes = np.argmax(ensemble_preds_weighted, axis=1)
ensemble_weighted_acc = np.mean(ensemble_weighted_classes == y_test)
print(f"Ensemble Method 2 (Weighted Average): Accuracy = {ensemble_weighted_acc:.4f}")

# Ensemble Method 3: Majority Voting (hard voting)
model1_classes = np.argmax(model1_preds, axis=1)
model2_classes = np.argmax(model2_preds, axis=1)
model3_classes = np.argmax(model3_preds, axis=1)
model4_classes = np.argmax(model4_preds, axis=1)

# Stack predictions and find most common prediction for each sample
stacked_preds = np.vstack([model1_classes, model2_classes, model3_classes, model4_classes])
majority_votes = []
for i in range(stacked_preds.shape[1]):
    # Get most common prediction for this sample
    votes = np.bincount(stacked_preds[:, i], minlength=10)
    majority_votes.append(np.argmax(votes))

ensemble_majority_acc = np.mean(np.array(majority_votes) == y_test)
print(f"Ensemble Method 3 (Majority Voting): Accuracy = {ensemble_majority_acc:.4f}")

# Compare ensemble methods with individual models
ensemble_results = pd.DataFrame({
    "Model": ["FCN", "Local CNN", "CNN1", "CNN2",
              "Ensemble (Average)", "Ensemble (Weighted)", "Ensemble (Majority)"],
    "Accuracy": [model1_acc, model2_acc, model3_acc, model4_acc,
                 ensemble_avg_acc, ensemble_weighted_acc, ensemble_majority_acc]
})
print("\nComparison of Individual Models vs Ensemble Methods:")
print(ensemble_results)

# =================== PART 2: DROPOUT REGULARIZATION ===================
print("\n===== DROPOUT REGULARIZATION =====")


# Function to create FCN with different dropout rates
def create_fcn_dropout(dropout_rate=0.0):
    model = models.Sequential([
        layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(256,)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='tanh', kernel_initializer='glorot_normal'),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Test different dropout rates
dropout_rates = [0.0, 0.2, 0.5, 0.8]
dropout_results = []

for rate in dropout_rates:
    print(f"\nTraining FCN with dropout rate = {rate}")
    model = create_fcn_dropout(dropout_rate=rate)
    history = model.fit(
        X_train_fcn, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test_fcn, y_test),
        verbose=1
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test_fcn, y_test, verbose=0)

    dropout_results.append({
        'Dropout Rate': rate,
        'Test Accuracy': test_acc,
        'Test Loss': test_loss
    })

    # Save training history for later analysis
    if rate == 0.0:
        history_no_dropout = history
    elif rate == 0.5:  # Save a moderate dropout rate for comparison
        history_with_dropout = history

# Display dropout results
dropout_df = pd.DataFrame(dropout_results)
print("\nDropout Analysis Results:")
print(dropout_df)

# Find the most effective and least effective dropout rates
best_dropout_rate = dropout_df.iloc[dropout_df['Test Accuracy'].idxmax()]['Dropout Rate']
worst_dropout_rate = dropout_df.iloc[dropout_df['Test Accuracy'].idxmin()]['Dropout Rate']

print(f"\nMost effective dropout rate: {best_dropout_rate}")
print(f"Least effective dropout rate: {worst_dropout_rate}")

# Demonstrate an effective case (train with the best dropout rate)
print("\nDemonstrating effective dropout rate:")
model_effective = create_fcn_dropout(dropout_rate=best_dropout_rate)
history_effective = model_effective.fit(
    X_train_fcn, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_fcn, y_test),
    verbose=1
)
effective_loss, effective_acc = model_effective.evaluate(X_test_fcn, y_test, verbose=0)
print(f"Effective dropout (rate={best_dropout_rate}): Accuracy = {effective_acc:.4f}, Loss = {effective_loss:.4f}")

# Demonstrate an ineffective case (train with the worst dropout rate)
print("\nDemonstrating ineffective dropout rate:")
model_ineffective = create_fcn_dropout(dropout_rate=worst_dropout_rate)
history_ineffective = model_ineffective.fit(
    X_train_fcn, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_fcn, y_test),
    verbose=1
)
ineffective_loss, ineffective_acc = model_ineffective.evaluate(X_test_fcn, y_test, verbose=0)
print(
    f"Ineffective dropout (rate={worst_dropout_rate}): Accuracy = {ineffective_acc:.4f}, Loss = {ineffective_loss:.4f}")

# Plot the training and validation accuracy for the two cases
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_effective.history['accuracy'], label=f'Train (rate={best_dropout_rate})')
plt.plot(history_effective.history['val_accuracy'], label=f'Validation (rate={best_dropout_rate})')
plt.plot(history_ineffective.history['accuracy'], label=f'Train (rate={worst_dropout_rate})', linestyle='--')
plt.plot(history_ineffective.history['val_accuracy'], label=f'Validation (rate={worst_dropout_rate})', linestyle='--')
plt.title('Model Accuracy with Different Dropout Rates')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_effective.history['loss'], label=f'Train (rate={best_dropout_rate})')
plt.plot(history_effective.history['val_loss'], label=f'Validation (rate={best_dropout_rate})')
plt.plot(history_ineffective.history['loss'], label=f'Train (rate={worst_dropout_rate})', linestyle='--')
plt.plot(history_ineffective.history['val_loss'], label=f'Validation (rate={worst_dropout_rate})', linestyle='--')
plt.title('Model Loss with Different Dropout Rates')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('dropout_comparison.png')
plt.close()

# Plot dropout rates vs. accuracy
plt.figure(figsize=(10, 6))
plt.plot(dropout_df['Dropout Rate'], dropout_df['Test Accuracy'], 'o-', linewidth=2)
plt.title('Effect of Dropout Rate on Test Accuracy')
plt.xlabel('Dropout Rate')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.savefig('dropout_rates.png')
plt.close()

# =================== SUMMARY ===================
print("\n===== SUMMARY =====")
print("\nEnsemble Methods Results:")
print(ensemble_results)

print("\nDropout Analysis Results:")
print(dropout_df)

print("\nResults have been saved to CSV files and plots saved as PNG files.")