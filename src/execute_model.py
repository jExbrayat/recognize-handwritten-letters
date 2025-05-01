import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("add_model.onnx")

x = np.array([1.0], dtype=np.float32)
y = np.array([2.0], dtype=np.float32)

inputs = {"x": x, "y": y}
outputs = sess.run(None, inputs)
print(outputs)
