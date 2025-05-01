import onnx
from onnx import helper, TensorProto

# Un petit modèle ONNX "Add"
node = helper.make_node("Add", ["x", "y"], ["z"])

graph = helper.make_graph(
    [node],
    "SimpleAddGraph",
    inputs=[
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    ],
    outputs=[
        helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
    ]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
onnx.save(model, "add_model.onnx")
