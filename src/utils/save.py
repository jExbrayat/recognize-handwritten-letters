import numpy as np
from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn

from src.utils.constants import data_folder


def save_model(model, x: np.ndarray, model_name: str) -> None:
    onnx_model = convert_sklearn(model, initial_types=[('float_input', FloatTensorType([None, x.shape[1]]))])

    with open(f"{data_folder}/models/{model_name}.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
