"""Saving models, data, results, etc."""

from pathlib import Path

import numpy as np
from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.pipeline import Pipeline

from src.utils.constants import data_folder


def save_model(
    model: Pipeline, x: np.ndarray, model_name: str, path: str = f"{data_folder}/models"
) -> None:
    """Convert a Scikit-Learn model to ONNX format and save it to disk.

    Parameters
    ----------
    model : Pipeline
        The trained Scikit-Learn model (e.g., Pipeline, RandomForestClassifier, etc.).
    x : np.ndarray
        Array of feature examples used to define the input shape of the model.
        Only the shape is used; the values themselves are ignored.
    model_name : str
        Output filename (without extension) used to name the ONNX file.
    path : str, optional
        Directory path where the file will be saved.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from src.data.load_data import load_sample_x_y
    >>>
    >>> x, y = load_sample_x_y()
    >>> model = DecisionTreeClassifier(max_depth=1).fit(x, y)
    >>> save_model(model, x, "decision-tree", path="tests/temp")
    """
    onnx_model = convert_sklearn(
        model, initial_types=[("float_input", FloatTensorType([None, x.shape[1]]))]
    )

    with Path(f"{path}/{model_name}.onnx").open("wb") as f:
        f.write(onnx_model.SerializeToString())
