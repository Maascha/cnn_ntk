"""NTK dispersion at initialization for one fixed input image (Figs. 1, 3, 4, 7).

For every width in --widths and every (sigma_w^2, depth) cell, draw --n-mc
independent initializations, compute Theta_0(x, x) for training image
--image, and estimate E[Theta_0^2] / E[Theta_0]^2 (bias-corrected estimator,
see dispersion.py). The result,
including every raw draw, is saved to --out.

    python src/run_dispersion.py --arch cnn --kernel-size 3 --image 0 --out caches/cnn_k3_img0.pt
    python src/run_dispersion.py --arch mlp --widths 144 216 288 432 576 864 1152 2304 --image 0 --out caches/mlp_img0.pt

Runs in float64 (Theta_0 overflows float32 in the chaotic regime). Resumable:
per-width progress is checkpointed next to --out; rerun the identical command
after an interruption.
"""

import argparse
import os
import time

import torch

import data as data_mod
from dispersion import sweep_grid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["cnn", "mlp"], default="cnn")
    p.add_argument("--widths", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96, 128, 256],
                   help="channel counts C (cnn) or hidden widths M (mlp)")
    p.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32])
    p.add_argument("--sigma-w2", type=float, nargs="+", default=[1.0, 3.44, 5.0])
    p.add_argument("--sigma-b2", type=float, default=1.0)
    p.add_argument("--n-mc", type=int, default=1000, help="initializations per cell")
    p.add_argument("--image", type=int, default=0,
                   help="index of the input image in the training set (see data.py)")
    p.add_argument("--kernel-size", type=int, default=3, help="cnn only")
    p.add_argument("--readout", choices=["gap", "flatten"], default="gap", help="cnn only")
    p.add_argument("--padding-mode", choices=["circular", "zeros"], default="circular",
                   help="cnn only")
    p.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output .pt file")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = data_mod.build_dataset(**data_mod.PAPER_DATA)
    x = ds["X_train"][args.image:args.image + 1].to(device=device, dtype=torch.float64)

    model_kw = dict(arch=args.arch, activation=args.activation)
    if args.arch == "cnn":
        model_kw.update(kernel_size=args.kernel_size, readout=args.readout,
                        padding_mode=args.padding_mode)
    print(f"device={device}  {model_kw}  image={args.image}  widths={args.widths}  "
          f"depths={args.depths}  sigma_w2={args.sigma_w2}  n_mc={args.n_mc}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    stem = os.path.splitext(args.out)[0]
    ratios, samples, t0 = [], [], time.time()
    for width in args.widths:
        ckpt = f"{stem}.ckpt_W{width}.pt"
        r, s = sweep_grid(x, args.sigma_w2, args.depths, checkpoint_path=ckpt, width=width,
                          sigma_b2=args.sigma_b2, n_mc=args.n_mc, device=device, **model_kw)
        ratios.append(r)
        samples.append({"ntk": s})
        print(f"width {width} done ({time.time() - t0:.0f}s)")
        header = "sigma_w^2 \\ L " + " ".join(f"{d:>9d}" for d in args.depths)
        print("  D_0 = ratio - 1\n  " + header)
        for sw, row in zip(args.sigma_w2, (r - 1).tolist()):
            print(f"  {sw:>12g} " + " ".join(f"{v:9.3g}" for v in row))

    torch.save({
        "panel_values": args.widths,
        "sigma_w2_values": args.sigma_w2,
        "depths": args.depths,
        "jackknife_ratio_grids": ratios,        # per width: [n_sigma, n_depth]
        "raw_samples": samples,                 # per width: {"ntk": [n_sigma, n_depth, n_mc]}
        "args": vars(args),
    }, args.out)
    for width in args.widths:
        ckpt = f"{stem}.ckpt_W{width}.pt"
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
