import torch
import onnx
from torch import Tensor

class ScanLikeModel(torch.nn.Module):
    def forward(self, state: Tensor, sequence: Tensor):
        outputs = []
        for i in range(sequence.shape[0]):
            elem = sequence[i]
            state = state + elem  # mise à jour de l'état
            outputs.append(state * 2)  # sortie
        return torch.stack(outputs), state

# Données d'entrée
state = torch.tensor(0.0)
sequence = torch.tensor([1.0, 2.0, 3.0])

model = ScanLikeModel()

# Export vers ONNX

onnx_model_path = "scan_explicit.onnx"
exported = torch.onnx.export(model, (state, sequence), dynamo=True)
exported.save(onnx_model_path)

model = onnx.load(onnx_model_path)

scan_nodes = [node for node in model.graph.node if node.op_type == "Scan"]

assert scan_nodes, "Aucun Scan trouvé dans le graphe !"
scan_node = scan_nodes[0]