"""Cross-check the three empirical-NTK implementations on small networks (float64):

  ntk.empirical_ntk_jacobian_contraction   explicit Jacobians (reference)
  ntk.empirical_ntk_ntk_vps                NTK-vector products (dispersion sweeps)
  kernels.gram                             Gram matrix for training, incl. its
                                           row-blocked and parameter-blocked paths

    python src/verify_ntk.py
"""

import torch

import kernels
from dispersion import build_model, init_variance_scaled_
from ntk import empirical_ntk_jacobian_contraction, empirical_ntk_ntk_vps

CASES = [
    dict(arch="mlp", width=16, depth=3),
    dict(arch="cnn", width=4, depth=3, kernel_size=3, readout="gap"),
    dict(arch="cnn", width=4, depth=2, kernel_size=5, readout="flatten"),
    dict(arch="cnn", width=4, depth=2, kernel_size=3, readout="gap", padding_mode="zeros"),
]


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    shape = (3, 6, 6)
    ok = True

    def check(name, a, b):
        nonlocal ok
        good = torch.allclose(a, b, rtol=1e-8, atol=1e-10)
        ok &= good
        rel = ((a - b).abs().max() / b.abs().max()).item()
        print(f"    {name:<34} rel. err {rel:.1e}  {'OK' if good else 'FAIL'}")

    for case in CASES:
        arch = case.pop("arch")
        model = build_model(arch, case.pop("width"), case.pop("depth"), shape, **case)
        init_variance_scaled_(model, sigma_w2=2.0, sigma_b2=1.0)
        params = {k: v.detach() for k, v in model.named_parameters()}
        x = torch.randn(6, *shape)
        x = x if arch == "cnn" else x.flatten(1)
        print(f"{arch} {case}")
        ref = empirical_ntk_jacobian_contraction(model, params, x, x)[..., 0, 0]
        check("ntk_vps vs jacobian contraction", empirical_ntk_ntk_vps(model, params, x, x, "trace"), ref)
        check("gram vs jacobian contraction", kernels.gram(model, x), ref)
        check("gram row_block=2, chunk_size=2", kernels.gram(model, x, row_block=2, chunk_size=2), ref)
        check("gram param_block=50", kernels.gram(model, x, param_block=50), ref)
        check("gram cross (x[:2], x)", kernels.gram(model, x[:2], x), ref[:2])

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
