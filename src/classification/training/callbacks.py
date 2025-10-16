"""Callbacks."""

import os

os.environ["KERAS_BACKEND"] = "torch"
from keras.src.callbacks import EarlyStopping, History


def early_stopping(monitor: str = "val_loss", patience: int = 25) -> EarlyStopping:
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode="min",
        restore_best_weights=True,
        start_from_epoch=50,
    )


def history() -> History:
    return History()
