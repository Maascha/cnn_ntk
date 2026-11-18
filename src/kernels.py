"""Gram matrix of the empirical NTK on a set of inputs, and the kernel-change metrics.

    J = [n, P] per-sample Jacobian (jacrev),   K = J1 J2^T

For a whole Gram matrix this is cheaper than NTK-vector products (n backward
passes and one matrix product). verify_ntk.py checks it against ntk.py.

Memory options (all give the same K up to float summation order):
  row_block    build J in blocks of rows
  chunk_size   vmap chunk size inside jacrev
  jac_device   keep the materialized J on another device (e.g. "cpu")
  param_block  accumulate K = sum_g J_g J_g^T over parameter groups of at most
               this many entries, so the full [n, P] Jacobian never exists
"""

import torch
from torch.func import functional_call, jacrev, vmap


def _params(model):
    return {k: v.detach() for k, v in model.named_parameters()}


def _param_groups(params, max_elems):
    """Split parameter tensors into consecutive groups of <= max_elems
    entries (a single tensor larger than that forms its own group)."""
    groups, cur, cur_n = [], [], 0
    for k, v in params.items():
        if cur and cur_n + v.numel() > max_elems:
            groups.append(cur)
            cur, cur_n = [], 0
        cur.append(k)
        cur_n += v.numel()
    if cur:
        groups.append(cur)
    return groups


def jacobian(model, X, chunk_size=None, out_device=None, row_block=None,
             param_names=None):
    """Per-sample Jacobian of the scalar output: [n, P], parameters in
    named_parameters() order.

    `row_block` bounds the peak on the MODEL's device: rows are built a block
    at a time and each block is moved to `out_device` before the next is
    built, so the high-water mark is row_block x P rather than n x P. Without
    it, out_device only decides where the finished matrix ends up.
    """
    allp = _params(model)
    if param_names is None:
        params, const = allp, {}
    else:
        want = set(param_names)
        params = {k: v for k, v in allp.items() if k in want}
        const = {k: v for k, v in allp.items() if k not in want}

    def f_single(p, x):
        # The non-differentiated tensors are closed over as constants, so
        # jacrev only builds gradients for `p`. That is what makes the
        # per-group peak P_group rather than P.
        return functional_call(model, {**const, **p}, (x.unsqueeze(0),)).squeeze()

    def block(xs):
        jac = vmap(jacrev(f_single), in_dims=(None, 0),
                   chunk_size=chunk_size)(params, xs)
        return torch.cat([j.reshape(xs.shape[0], -1) for j in jac.values()], dim=1)

    n = X.shape[0]
    if row_block is None or row_block >= n:
        flat = block(X)
        return flat.to(out_device) if out_device is not None else flat

    parts = []
    for lo in range(0, n, row_block):
        blk = block(X[lo:lo + row_block])
        parts.append(blk.to(out_device) if out_device is not None else blk)
        del blk
    return torch.cat(parts, dim=0)


def gram(model, X1, X2=None, *, row_block=None, chunk_size=None,
         jac_device=None, param_block=None):
    """K[i, j] = <grad_theta f(X1_i), grad_theta f(X2_j)>, shape [n1, n2].

    X2=None gives the symmetric Gram matrix K(X1, X1), computed from a single
    Jacobian. `row_block` bounds how many rows of J exist at once; None means
    one pass. `chunk_size` is passed to vmap and bounds the peak inside the
    jacrev itself. `jac_device` ("cpu") parks the materialized Jacobian off
    the GPU.
    """
    device = X1.device
    symmetric = X2 is None
    X2 = X1 if symmetric else X2
    n1, n2 = X1.shape[0], X2.shape[0]

    if param_block:
        # K = J J^T = sum over parameter groups of J_g J_g^T, so the kernel
        # can be accumulated without the full [n, P] Jacobian ever existing.
        # Memory then scales with the LARGEST GROUP, not the total parameter
        # count. Costs one backward pass per group.
        groups = _param_groups(_params(model), param_block)
        out = torch.zeros(n1, n2, dtype=X1.dtype, device=device)
        for names in groups:
            j1 = jacobian(model, X1, chunk_size=chunk_size,
                          out_device=jac_device, row_block=row_block,
                          param_names=names)
            j2 = j1 if symmetric else jacobian(
                model, X2, chunk_size=chunk_size, out_device=jac_device,
                row_block=row_block, param_names=names)
            out += (j1 @ j2.T).to(device)
            del j1, j2
        return out

    if row_block is None:
        j1 = jacobian(model, X1, chunk_size=chunk_size, out_device=jac_device)
        j2 = j1 if symmetric else jacobian(model, X2, chunk_size=chunk_size,
                                           out_device=jac_device)
        return (j1 @ j2.T).to(device)

    # Blocked: J2 is materialized once (possibly on the CPU), then each row
    # block of J1 is contracted against it.
    j2 = jacobian(model, X2, chunk_size=chunk_size, out_device=jac_device,
                  row_block=row_block)
    out = torch.empty(n1, n2, dtype=j2.dtype, device=device)
    for lo in range(0, n1, row_block):
        hi = min(lo + row_block, n1)
        if symmetric:
            j1_blk = j2[lo:hi]
        else:
            j1_blk = jacobian(model, X1[lo:hi], chunk_size=chunk_size,
                              out_device=jac_device, row_block=row_block)
        out[lo:hi] = (j1_blk @ j2.T).to(device)
        del j1_blk
    return out


# ---------------------------------------------------------------------------
# Scalar statistics of a Gram matrix
# ---------------------------------------------------------------------------

def kernel_change(K_t, K_0):
    """Kernel movement relative to initialization, as a dict.

      delta = ||K_t - K_0||_F / ||K_0||_F              relative change Delta_t (paper)
      d     = 1 - <K_t, K_0>_F / (||K_t||_F ||K_0||_F)  rotation R_t (paper)
      s     = log(||K_t||_F / ||K_0||_F)               change of overall scale
    """
    n0 = K_0.norm()
    nt = K_t.norm()
    cos = (K_t * K_0).sum() / (nt * n0).clamp_min(1e-300)
    return {
        "delta": float((K_t - K_0).norm() / n0),
        "s": float(torch.log(nt / n0)),
        "d": float(1.0 - cos),
        "K_norm": float(nt),
    }
