"""Data preprocessing."""

import numpy as np


def preprocess(x: np.ndarray) -> np.ndarray:
    """Preprocess image data.

    Parameters
    ----------
    x : np.ndarray
        Input array containing raw pixel values, in the range [0, 255].

    Returns
    -------
    np.ndarray
        Preprocessed array of images.
    """
    x = x.astype(np.float32)
    return x / 255.0
