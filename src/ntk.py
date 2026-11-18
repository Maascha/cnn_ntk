"""Empirical NTK Theta(x1, x2) = <grad_theta f(x1), grad_theta f(x2)>, two ways.

1. `empirical_ntk_jacobian_contraction`: explicit per-sample Jacobians (jacrev),
   contracted over the parameters. Simple; used as the reference.
2. `empirical_ntk_ntk_vps`: NTK-vector products (a vjp composed with a jvp),
   never materializes the Jacobian. Used for the dispersion sweeps.

Both follow the PyTorch NTK tutorial
(https://docs.pytorch.org/tutorials/intermediate/neural_tangent_kernels.html).
verify_ntk.py checks that they agree.
"""

import torch
from torch.func import functional_call, jacrev, jvp, vjp, vmap


def _fnet_single(model, params, x):
    return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)


def empirical_ntk_jacobian_contraction(model, params, x1, x2):
    """Theta of shape [N, M, K, K] (K = output dim)."""
    def fnet_single(params, x):
        return _fnet_single(model, params, x)

    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1.values()]
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2.values()]
    return torch.stack(
        [torch.einsum("Naf,Mbf->NMab", j1, j2) for j1, j2 in zip(jac1, jac2)]
    ).sum(0)


def empirical_ntk_ntk_vps(model, params, x1, x2, compute="full"):
    """Theta via NTK-vector products. compute: "full" -> [N, M, K, K],
    "trace" -> [N, M], "diagonal" -> [N, M, K]."""

    def fnet_single(params, x):
        return _fnet_single(model, params, x)

    def get_ntk(x1, x2):
        def func_x1(params):
            return fnet_single(params, x1)

        def func_x2(params):
            return fnet_single(params, x2)

        out1, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            vjps = vjp_fn(vec)
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        basis = torch.eye(out1.numel(), dtype=out1.dtype,
                          device=out1.device).view(out1.numel(), -1)
        return vmap(get_ntk_slice)(basis)

    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    if compute == "full":
        return result
    if compute == "trace":
        return torch.einsum("NMKK->NM", result)
    if compute == "diagonal":
        return torch.einsum("NMKK->NMK", result)
    raise ValueError(f"unknown compute mode: {compute}")
