import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class CNN(nn.Module):
    """`depth` conv layers with `channels` channels each, stride 1,
    padding="same", no pooling between layers, followed by a linear readout.

    readout="gap": global average pooling over H x W, then Linear(C, out_dim)
    (the architecture of the paper). readout="flatten": Linear(C*H*W, out_dim).
    padding_mode="circular" (paper) or "zeros".
    """

    def __init__(
        self,
        in_channels,
        channels,
        depth,
        input_hw,
        kernel_size=3,
        out_dim=1,
        activation="tanh",
        bias=True,
        padding_mode="circular",
        readout="gap",
    ):
        super().__init__()
        if readout not in ("flatten", "gap"):
            raise ValueError(f"readout must be 'flatten' or 'gap', got {readout!r}")
        self.readout_kind = readout
        act_cls = _ACTIVATIONS[activation]

        in_ch = [in_channels] + [channels] * depth
        layers = []
        for c_in, c_out in zip(in_ch[:-1], in_ch[1:]):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size, stride=1, padding="same",
                                    bias=bias, padding_mode=padding_mode))
            layers.append(act_cls())
        self.conv = nn.Sequential(*layers)

        h, w = input_hw
        fan_in = in_ch[-1] if readout == "gap" else in_ch[-1] * h * w
        self.readout = nn.Linear(fan_in, out_dim, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=(-2, -1)) if self.readout_kind == "gap" else x.flatten(1)
        return self.readout(x)
