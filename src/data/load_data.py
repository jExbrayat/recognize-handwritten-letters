"""Data loading."""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils.constants import data_folder


def load_data(path: str) -> pd.DataFrame:
    """Load data

    Examples
    --------
    >>> sample_data = load_data(f"{data_folder}/raw/sample-uniform-distribution.parquet")
    """
    data = pd.read_parquet(path, engine="pyarrow")
    return data


def load_full_data() -> pd.DataFrame:
    """Load full data.

    Examples
    --------
    >>> data = load_full_data()
    """
    data = load_data(f"{data_folder}/raw/original-data.parquet")
    return data


def load_sample_data() -> pd.DataFrame:
    """Load sample data.

    Examples
    --------
    >>> sample_data = load_sample_data()
    """
    sample_data = load_data(f"{data_folder}/raw/sample-uniform-distribution.parquet")
    return sample_data


def get_x_y(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Get x and y, where x is the input data and y the labels to predict.

    Examples
    --------
    >>> sample_data = load_sample_data()
    >>> x, y = get_x_y(sample_data)
    """
    x: np.ndarray = data.iloc[:, 1:].to_numpy()
    y: np.ndarray = data.iloc[:, 0].to_numpy()
    return x, y


def load_sample_x_y() -> tuple[np.ndarray, np.ndarray]:
    """Load sample x and y.

    Examples
    --------
    >>> x, y = load_sample_x_y()
    """
    sample_data = load_sample_data()
    x, y = get_x_y(sample_data)

    return x, y


def load_sample_train_test(
    test_size: float = 0.2, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sample train test.

    Examples
    --------
    >>> x_train, x_test, y_train, y_test = load_sample_train_test()
    """
    x, y = load_sample_x_y()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, random_state=random_state, test_size=test_size
    )

    return x_train, x_test, y_train, y_test


def convert_to_torch_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
) -> DataLoader:
    """Turn x and y into torch dataset.

    Examples
    --------
    >>> x, y = load_sample_x_y()
    >>> data = convert_to_torch_dataset(x, y)
    """
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    x_tensor = x_tensor.view(-1, 1, 28, 28)

    dataset = TensorDataset(x_tensor, y_tensor)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader
