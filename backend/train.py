# =============================================================================
#  train.py  —  CNN Training Script
#  Handwritten Character Recognition  |  MNIST Dataset
#
#  Run once before starting the API server:
#      cd backend
#      python train.py
# =============================================================================

import os, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.h5")

print("\n" + "="*55)
print("  Handwritten Digit Recognition — CNN Trainer")
print("="*55)

# 1 ── Load MNIST ──────────────────────────────────────────────────────────────
print("\n[1/4]  Loading MNIST (auto-download ~11 MB on first run) …")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1,  28, 28, 1).astype("float32") / 255.0
y_train_enc = to_categorical(y_train, 10)
y_test_enc  = to_categorical(y_test,  10)
print(f"       Train: {len(x_train):,}  |  Test: {len(x_test):,}")

# 2 ── Build model ─────────────────────────────────────────────────────────────
print("\n[2/4]  Building CNN …")

model = models.Sequential(name="DigitCNN")
model.add(layers.Input(shape=(28, 28, 1)))

# Block 1
model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2))
model.add(layers.Dropout(0.25))

# Block 2
model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2))
model.add(layers.Dropout(0.25))

# Block 3
model.add(layers.Conv2D(128, 3, activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))

# Classifier
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

# 3 ── Train ───────────────────────────────────────────────────────────────────
print("\n[3/4]  Training …")
os.makedirs(MODEL_DIR, exist_ok=True)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    x_train, y_train_enc,
    epochs=20, batch_size=128, validation_split=0.1,
    callbacks=[
        EarlyStopping(monitor="val_accuracy", patience=4,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                         patience=2, min_lr=1e-6, verbose=1),
    ],
    verbose=1,
)

# 4 ── Evaluate ────────────────────────────────────────────────────────────────
print("\n[4/4]  Evaluating …")
loss, acc = model.evaluate(x_test, y_test_enc, verbose=0)
print("\n" + "="*55)
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  Test Loss     : {loss:.4f}")
print(f"  Model saved   → {os.path.abspath(MODEL_PATH)}")
print("\n  Next: uvicorn main:app --reload --port 8000")
print("="*55 + "\n")
