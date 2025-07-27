"""Deep learning models."""

import os

from src.utils.constants import img_shape_flat

os.environ["KERAS_BACKEND"] = "torch"

from keras.src.layers import Dense, Input
from keras.src.metrics import F1Score
from keras.src.models import Sequential
from keras.src.optimizers import Adam


def sequential_model() -> Sequential:
    model = Sequential()
    model.add(Input(shape=img_shape_flat))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(26, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy", F1Score()],
    )

    return model
