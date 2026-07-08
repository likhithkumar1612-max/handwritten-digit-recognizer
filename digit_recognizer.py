"""
Handwritten Digit Recognizer
-----------------------------
Trains a Convolutional Neural Network (CNN) to recognize handwritten digits
(0-9) using the classic digits dataset (8x8 grayscale images, a standard
scikit-learn-bundled subset in the spirit of MNIST).

No internet download required — the dataset ships with scikit-learn.

Author: <your name>
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots to file
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_data():
    """Load and preprocess the digits dataset."""
    digits = load_digits()
    X = digits.images  # shape: (n_samples, 8, 8)
    y = digits.target  # shape: (n_samples,)

    # Normalize pixel values to [0, 1]
    X = X.astype("float32") / 16.0

    # Reshape for CNN input: (samples, height, width, channels)
    X = X.reshape(-1, 8, 8, 1)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_model():
    """Build a small CNN suitable for 8x8 digit images."""
    model = keras.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history, out_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training history plot to {out_path}")


def plot_sample_predictions(model, X_test, y_test, out_path="sample_predictions.png"):
    preds = np.argmax(model.predict(X_test[:10], verbose=0), axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
        ax.set_title(f"True: {y_test[i]}\nPred: {preds[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved sample predictions to {out_path}")


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    print("\nBuilding model...")
    model = build_model()
    model.summary()

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=25,
        batch_size=32,
        verbose=2,
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    plot_training_history(history)
    plot_sample_predictions(model, X_test, y_test)

    model.save("digit_recognizer_model.keras")
    print("\nModel saved to digit_recognizer_model.keras")


if __name__ == "__main__":
    main()
