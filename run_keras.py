"""Keras (TensorFlow) feed-forward network. Usage: python run_keras.py credit|german"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras
import numpy as np
from keras import layers

from common import Timer, evaluate, get_split, pos_weight, save_result

ds = sys.argv[1]
keras.utils.set_random_seed(42)
X_tr, X_te, y_tr, y_te, _ = get_split(ds)
model = keras.Sequential([
    layers.Input(shape=(X_tr.shape[1],)),
    layers.Dense(64, activation="relu"), layers.Dropout(0.2),
    layers.Dense(32, activation="relu"), layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
w = pos_weight(y_tr)
w = min(w, 20.0)  # cap the class weight so training stays stable on the very imbalanced credit data
with Timer() as t:
    model.fit(X_tr, y_tr, epochs=20, batch_size=256, validation_split=0.1, verbose=0,
              class_weight={0: 1.0, 1: w},
              callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
save_result("Keras", ds, evaluate(y_te, model.predict(X_te, verbose=0).ravel()), t.seconds, loc=12,
            notes="2 hidden layers (64, 32) + dropout; class-weighted loss; early stopping")
