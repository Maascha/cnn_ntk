"""Run configuration and model construction for the training experiments."""

import dataclasses
import math
from dataclasses import dataclass

import torch

from dispersion import build_model as _build, init_variance_scaled_

ARCHS = ("cnn", "mlp")


@dataclass(frozen=True)
class RunConfig:
    """One grid cell (everything except the init seed)."""

    arch: str = "cnn"
    width: int = 64             # channel count C (cnn) or hidden width M (mlp)
    depth: int = 8              # number of conv / hidden layers L
    kernel_size: int = 3        # cnn only
    sigma_w2: float = 1.0
    sigma_b2: float = 1.0
    activation: str = "tanh"
    readout: str = "gap"        # cnn only: "gap" (paper) or "flatten"
    padding_mode: str = "circular"

    # data (data.build_dataset)
    hw: int = 16
    n_train: int = 2048
    n_test: int = 2000
    classes: tuple = (0, 6)
    data_seed: int = 0
    input_scale: str = "sqrt_dim"

    # optimization: full-batch GD, eta = c_lr / lambda_max(Theta_0(X, X))
    c_lr: float = 1.0
    max_steps: int = 100_000
    target_mse: float = 0.1     # stop at the first step with training MSE <= target_mse
    kernel_probe: int = 256     # Theta_t is measured on the first P training points

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.kernel_probe % 2 or not 0 < self.kernel_probe <= self.n_train:
            raise ValueError(f"kernel_probe must be even and in (0, n_train], "
                             f"got {self.kernel_probe}")

    @property
    def lam(self):
        """lambda = L / C for the CNN; L / (M / k^2) = k^2 L / M for the MLP
        when compared against a CNN with kernel size k (paper: M = 9C)."""
        return self.depth / self.width

    def slug(self):
        name = (f"mlp_M{self.width}" if self.arch == "mlp"
                else f"cnn_k{self.kernel_size}_C{self.width}_{self.readout}")
        sw = f"{self.sigma_w2:g}".replace(".", "p")
        return f"{name}_L{self.depth}_sw{sw}_{self.activation}"

    def to_dict(self):
        return dataclasses.asdict(self) | {"slug": self.slug()}


def build_model(cfg, device=None):
    model = _build(cfg.arch, cfg.width, cfg.depth, (3, cfg.hw, cfg.hw),
                   kernel_size=cfg.kernel_size, activation=cfg.activation,
                   readout=cfg.readout, padding_mode=cfg.padding_mode)
    return model if device is None else model.to(device)


def init_model(cfg, seed, device=None):
    """Seed, build, variance-scale. In float32 runs the init is drawn in
    float64 and rounded, so a given seed gives the same network in both."""
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed)
        model = build_model(cfg, device=device)
        init_variance_scaled_(model, sigma_w2=cfg.sigma_w2, sigma_b2=cfg.sigma_b2)
    finally:
        torch.set_default_dtype(dtype)
    return model.to(dtype)


def prepare_inputs(cfg, x):
    """Image batch -> model input (the MLP sees the flattened image)."""
    return x.flatten(1) if cfg.arch == "mlp" else x


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def suggest_row_block(cfg, n_points, budget_bytes=2 * 1024 ** 3):
    """Largest power-of-two Jacobian row block whose [rows, P] slab fits in budget."""
    per_row = max(1, n_params(build_model(cfg)) * 8)
    rows = max(1, int(budget_bytes // per_row))
    return min(n_points, 1 << int(math.floor(math.log2(rows))))
