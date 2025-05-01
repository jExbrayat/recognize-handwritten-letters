import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


@onnx_impl(torch.ops.higher_order.scan)
def scan():
    pass

import torch.nn as nn

class SimpleScanModel(nn.Module):
    def forward(self, x):
        # x est un vecteur 1D, par exemple : tensor([1., 2., 3., 4.])
        init = torch.tensor(0.0)

        def step(state, input):
            new_state = state + input
            return new_state, new_state

        final, outputs = torch.scan(step, init, x)
        return outputs  # somme cumulative des éléments

# Exemple d'utilisation
x = torch.tensor([1., 2., 3., 4.])
model = SimpleScanModel()
out = model(x)

print(out)  # tensor([1., 3., 6., 10.])

