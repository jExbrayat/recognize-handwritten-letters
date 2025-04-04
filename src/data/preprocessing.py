"""Data preprocessing."""

import numpy as np


def preprocess(x: np.ndarray) -> np.ndarray:
    return x / 255.0
