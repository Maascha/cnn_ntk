"""NTK dispersion at initialization.

For a fixed input x, draw n_mc independent initializations, compute
Theta_0(x, x) for each, and estimate the dispersion ratio
E[Theta_0(x,x)^2] / E[Theta_0(x,x)]^2 with the bias-corrected (leave-one-out)
estimator of Seleznova & Kutyniok (ICML 2022, Appendix C.3), implemented in
jackknife_ratio_batched below. The excess dispersion of the paper is D_0(x) = ratio - 1.

Initialization is standard parametrization: W ~ N(0, sigma_w^2 / fan_in),
b ~ N(0, sigma_b^2), with fan_in = C_in * k * k for a conv layer.
"""

import os

import torch
import torch.nn as nn

from cnn import CNN
from mlp import MLP
from ntk import empirical_ntk_ntk_vps


def init_variance_scaled_(model, sigma_w2, sigma_b2=0.0):
    """In place: W ~ N(0, sigma_w2 / fan_in), b ~ N(0, sigma_b2) for every
    Linear / Conv2d. fan_in = in_features, or C_in * k_h * k_w for a conv."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            else:
                fan_in = module.weight.shape[1]
            nn.init.normal_(module.weight, mean=0.0, std=(sigma_w2 / fan_in) ** 0.5)
            if module.bias is not None:
                nn.init.normal_(module.bias, mean=0.0, std=sigma_b2 ** 0.5)


def build_model(arch, width, depth, input_shape, *, kernel_size=3, activation="tanh",
                readout="gap", padding_mode="circular"):
    """arch "cnn": `width` = channel count C; arch "mlp": `width` = hidden width M
    (the input is flattened)."""
    c, h, w = input_shape
    if arch == "mlp":
        return MLP(c * h * w, width, depth, out_dim=1, activation=activation)
    return CNN(c, width, depth, (h, w), kernel_size=kernel_size, out_dim=1,
               activation=activation, padding_mode=padding_mode, readout=readout)


def sample_theta0(x, *, arch, width, depth, sigma_w2, sigma_b2, n_mc, device, **model_kw):
    """n_mc samples of Theta_0(x, x) over independent initializations, for one
    fixed input x of shape [1, C, H, W]."""
    model = build_model(arch, width, depth, tuple(x.shape[1:]), **model_kw).to(device)
    if arch == "mlp":
        x = x.flatten(1)
    thetas = torch.empty(n_mc, device=device)
    for i in range(n_mc):
        init_variance_scaled_(model, sigma_w2, sigma_b2)
        params = {k: v.detach() for k, v in model.named_parameters()}
        thetas[i] = empirical_ntk_ntk_vps(model, params, x, x, compute="trace").reshape(())
    return thetas


def jackknife_ratio_batched(thetas):
    """Leave-one-out estimator of E[Theta^2] / E[Theta]^2 along the last axis:
    (..., n_mc) -> (...).

    For each held-out draw k, theta_k^2 is divided by an unbiased U-statistic
    estimate of E[Theta]^2 built from the other n_mc - 1 draws, and the
    quotients are averaged. Unlike the plug-in ratio mean(theta^2)/mean(theta)^2,
    which is bounded above by n_mc, this has no n_mc ceiling. This is the
    bias-corrected estimator of Seleznova & Kutyniok (ICML 2022, Appendix C.3;
    reference implementation: github.com/mselezniova/ntk_beyond_limit).
    """
    n = thetas.shape[-1]
    if n < 3:
        raise ValueError(f"jackknife ratio needs n_mc >= 3, got {n}")
    thetas = thetas / thetas.abs().amax(dim=-1, keepdim=True)   # scale-invariant
    total = thetas.sum(dim=-1, keepdim=True)
    sq_total = thetas.pow(2).sum(dim=-1, keepdim=True)
    m = n - 1
    loo_mean = (total - thetas) / m
    loo_sq_mean = (sq_total - thetas.pow(2)) / m
    denom = (m / (m - 1)) * loo_mean.pow(2) - loo_sq_mean / (m - 1)
    return (thetas.pow(2) / denom).mean(dim=-1)


def sweep_grid(x, sigma_w2_values, depths, checkpoint_path=None, **kwargs):
    """Sample every (sigma_w2, depth) cell for one width. Returns
    (ratio [n_sigma, n_depth], samples [n_sigma, n_depth, n_mc]).

    With checkpoint_path, progress is saved after every cell and an
    interrupted run resumes from it (same arguments required)."""
    shape = (len(sigma_w2_values), len(depths))
    fingerprint = repr((list(sigma_w2_values), list(depths),
                        sorted((k, str(v)) for k, v in kwargs.items())))
    ratio = torch.full(shape, float("nan"))
    samples = torch.full((*shape, kwargs["n_mc"]), float("nan"))
    done = torch.zeros(shape, dtype=torch.bool)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        if ckpt["fingerprint"] == fingerprint:
            ratio, samples, done = ckpt["ratio"], ckpt["samples"], ckpt["done"]
            print(f"resuming: {int(done.sum())}/{done.numel()} cells done")
        else:
            print(f"ignoring stale checkpoint {checkpoint_path}")

    for i, sigma_w2 in enumerate(sigma_w2_values):
        for j, depth in enumerate(depths):
            if done[i, j]:
                continue
            thetas = sample_theta0(x, sigma_w2=sigma_w2, depth=depth, **kwargs)
            samples[i, j] = thetas.detach().to("cpu", torch.float64)
            ratio[i, j] = jackknife_ratio_batched(samples[i, j]).item()
            done[i, j] = True
            if checkpoint_path:
                tmp = checkpoint_path + ".tmp"
                torch.save({"fingerprint": fingerprint, "ratio": ratio,
                            "samples": samples, "done": done}, tmp)
                os.replace(tmp, checkpoint_path)
    return ratio, samples
