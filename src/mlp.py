import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """`depth` hidden layers of width `width`, followed by a linear readout."""

    def __init__(self, in_dim, width, depth, out_dim=1, activation="tanh", bias=True):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]

        dims = [in_dim] + [width] * depth
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out, bias=bias))
            layers.append(act_cls())
        layers.append(nn.Linear(dims[-1], out_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
