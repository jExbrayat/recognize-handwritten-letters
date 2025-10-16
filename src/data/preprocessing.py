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

    Examples
    --------
    >>> from src.data.load_data import load_sample_x_y
    >>> x, y = load_sample_x_y()
    >>> x_processed = preprocess(x)
    """
    x = x.astype(np.float32)
    return x / 255.0
