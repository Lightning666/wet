"""STM32-friendly LSTM example built with TensorFlow 2.x / Keras.

This script creates a compact LSTM model for multivariate water-quality time-series
classification and is intentionally constrained for easier deployment with STM32
microcontrollers via X-CUBE-AI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ==============================
# 1) Fixed data definition
# ==============================
# The deployment target expects a single time-series sample at inference time,
# but training can still use any batch size.
TIME_STEPS = 64
FEATURES = 8
FEATURE_NAMES = [
    "pH",
    "Turbidity (NTU)",
    "Temperature (°C)",
    "DO (mg/L)",
    "BOD (mg/L)",
    "Lead (mg/L)",
    "Mercury (mg/L)",
    "Arsenic (mg/L)",
]

# Example task choice:
# - classification: output uses softmax (good for quality-level / risk-level labels)
# - regression: switch the final layer to linear and update the loss accordingly
NUM_CLASSES = 4
TRAIN_SAMPLES = 256
VAL_SAMPLES = 64
EPOCHS = 3
BATCH_SIZE = 16
MODEL_PATH = Path("lstm_stm32_model.h5")
SEED = 42

# Keep training deterministic for a lightweight reproducible demo.
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model() -> keras.Model:
    """Build a compact model that stays well below STM32 flash limits.

    Design choices for STM32 / X-CUBE-AI compatibility:
    - Only 2 LSTM layers are used.
    - LSTM units are 32 and 16, both <= 64.
    - Dense layers are small (16 and output layer).
    - Default LSTM activations are preserved: tanh + sigmoid recurrent activation.
    - No BatchNorm, Embedding, custom RNN cells, unroll, or go_backwards.
    - A small L2 regularizer helps keep weights in a moderate range for later int8
      quantization.
    """
    inputs = keras.Input(shape=(TIME_STEPS, FEATURES), name="sensor_window")

    x = layers.LSTM(
        units=32,
        return_sequences=True,
        dropout=0.10,  # Training-only; ignored during inference/deployment.
        recurrent_dropout=0.0,
        kernel_regularizer=regularizers.l2(1e-5),
        recurrent_regularizer=regularizers.l2(1e-5),
        name="lstm_1",
    )(inputs)

    x = layers.LSTM(
        units=16,
        return_sequences=False,
        dropout=0.10,
        recurrent_dropout=0.0,
        kernel_regularizer=regularizers.l2(1e-5),
        recurrent_regularizer=regularizers.l2(1e-5),
        name="lstm_2",
    )(x)

    x = layers.Dense(
        units=16,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-5),
        name="dense_1",
    )(x)

    # Softmax output for classification. For regression, replace this layer with:
    # layers.Dense(1, activation="linear", name="regression_output")
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="stm32_compact_lstm")
    return model


def print_parameter_budget(model: keras.Model) -> None:
    """Print total parameter count and approximate storage footprint in KB.

    float32 uses 4 bytes per parameter during standard training/export.
    The 200 KB target corresponds to about 50,000 parameters.
    """
    total_params = model.count_params()
    total_kb = total_params * 4 / 1024
    within_budget = total_params <= 50_000

    print("\nFeature list:")
    for idx, name in enumerate(FEATURE_NAMES, start=1):
        print(f"  {idx}. {name}")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Approximate float32 parameter size: {total_kb:.2f} KB")
    print(f"Within ~200 KB STM32 budget: {within_budget}")


def generate_demo_data(samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random demo data only to verify the training flow.

    The values are normalized to a compact range to emulate quantization-friendly
    training data and avoid extreme activation/weight growth.
    """
    x = np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(samples, TIME_STEPS, FEATURES),
    ).astype(np.float32)

    labels = np.random.randint(0, NUM_CLASSES, size=(samples,))
    y = keras.utils.to_categorical(labels, num_classes=NUM_CLASSES).astype(np.float32)
    return x, y


def main() -> None:
    # Build the model with flexible training batch size.
    # For deployment on STM32, inference is typically run with batch size = 1.
    model = build_model()

    optimizer = keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0,  # Gradient clipping helps avoid extreme weights.
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Required by the task: print network structure.
    model.summary()
    print_parameter_budget(model)

    # Random demo data: enough to validate the end-to-end workflow.
    x_train, y_train = generate_demo_data(TRAIN_SAMPLES)
    x_val, y_val = generate_demo_data(VAL_SAMPLES)

    print("\nStart demo training...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nValidation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Completed epochs: {len(history.history['loss'])}")

    # Save as a single H5 file for easy handoff to STM32 workflows.
    model.save(MODEL_PATH, include_optimizer=True)
    print(f"\nModel saved to: {MODEL_PATH.resolve()}")

    # Optional next step (not executed here to preserve X-CUBE-AI compatibility):
    # Convert the trained .h5 model to TFLite and apply post-training int8 quantization
    # using a representative dataset before importing into embedded workflows.


if __name__ == "__main__":
    main()
