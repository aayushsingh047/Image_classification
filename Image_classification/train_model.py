"""
Image Classifier Training Script using TensorFlow/Keras
Trains a model on CIFAR-10 dataset and saves it for Streamlit app
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

print(f"Training set size: {x_train.shape}")
print(f"Test set size: {x_test.shape}")

# Build CNN model
print("\nBuilding CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Train the model
print("\nTraining model...")
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model_path = 'models/cifar10_classifier.h5'
model.save(model_path)
print(f"\nModel saved to {model_path}")

# Save class names to a file for the Streamlit app
import json
with open('models/class_names.json', 'w') as f:
    json.dump(class_names, f)
print("Class names saved to models/class_names.json")
