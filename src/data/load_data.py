from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.constants import data_folder


def load_sample_data() -> pd.DataFrame:
    """Load sample data.

    >>> df = load_sample_data()
    """
    df = pd.read_csv(f"{data_folder}/raw/sample-dataset.csv", header=None)
    return df


def get_x_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Get x and y, where x is the input data and y the labels to predict.

    >>> df = load_sample_data()
    >>> x, y = get_x_y(df)
    """
    x: np.ndarray = df.iloc[:, 1:].to_numpy()
    y: np.ndarray = df.iloc[:, 0].to_numpy()
    return x, y


def load_sample_x_y() -> Tuple[np.ndarray, np.ndarray]:
    """Load sample x and y.

    >>> x, y = load_sample_x_y()"""
    df = load_sample_data()
    x, y = get_x_y(df)

    return x, y
