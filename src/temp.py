import torch
import onnx

onnx_model_path = "exported_model.onnx"

class SimpleModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y


model = SimpleModel()
x = torch.Tensor([42])
y = torch.Tensor([33])


exported = torch.onnx.export(model, (x, y), dynamo=True)
exported.save(onnx_model_path)


onnx_model = onnx.load(onnx_model_path)
print("---------------------ok")
onnx.checker.check_model(onnx_model)
