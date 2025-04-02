import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.constants import data_folder


def load_sample_data() -> pd.DataFrame:
    """Load sample data.

    >>> df = load_sample_data()
    """
    df = pd.read_parquet(
        f"{data_folder}/raw/sample-uniform-distribution.parquet", engine="pyarrow"
    )
    return df


def get_x_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Get x and y, where x is the input data and y the labels to predict.

    >>> df = load_sample_data()
    >>> x, y = get_x_y(df)
    """
    x: np.ndarray = df.iloc[:, 1:].to_numpy()
    y: np.ndarray = df.iloc[:, 0].to_numpy()
    return x, y


def load_sample_x_y() -> tuple[np.ndarray, np.ndarray]:
    """Load sample x and y.

    >>> x, y = load_sample_x_y()
    """
    df = load_sample_data()
    x, y = get_x_y(df)

    return x, y


def load_sample_train_test(
    test_size: float = 0.2, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sample train test.

    >>> x_train, x_test, y_train, y_test = load_sample_train_test()
    """
    x, y = load_sample_x_y()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, random_state=random_state, test_size=test_size
    )

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test
