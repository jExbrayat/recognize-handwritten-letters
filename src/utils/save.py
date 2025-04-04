from pathlib import Path

import numpy as np
from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.pipeline import Pipeline

from src.utils.constants import data_folder


def save_model(model: Pipeline, x: np.ndarray, model_name: str) -> None:
    onnx_model = convert_sklearn(
        model, initial_types=[("float_input", FloatTensorType([None, x.shape[1]]))]
    )

    with Path(f"{data_folder}/models/{model_name}.onnx").open("wb") as f:
        f.write(onnx_model.SerializeToString())
