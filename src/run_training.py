"""Training sweep: every (sigma_w^2, width, depth) cell x every seed, one .pt per run
in <out-dir>/runs/. Resumable: rerun the identical command and finished runs are
skipped.

    # CNN, k = 3 (Figs. 2, 3)
    python src/run_training.py --arch cnn --widths 16 32 --depths 2 4 8 --sigma-w2 1.0 3.44 5.0 \
        --out-dir results/cnn_k3
    # MLP at matched fan-in M = 9C (appendix)
    python src/run_training.py --arch mlp --widths 144 288 --depths 2 4 8 --out-dir results/mlp

The paper's runs used --float32 (TF32 disabled; the init is still drawn in float64).
"""

import argparse
import itertools
import json
import os
import sys
import time

import torch

import data as data_mod
from models import RunConfig, build_model, n_params, suggest_row_block
from train import run_training


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arch", choices=("cnn", "mlp"), default="cnn")
    ap.add_argument("--widths", nargs="+", type=int, required=True,
                    help="channel counts C (cnn) or hidden widths M (mlp)")
    ap.add_argument("--depths", nargs="+", type=int, required=True)
    ap.add_argument("--sigma-w2", nargs="+", type=float, default=[1.0, 3.44, 5.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--readout", choices=("gap", "flatten"), default="gap")
    ap.add_argument("--padding-mode", choices=("circular", "zeros"), default="circular")
    ap.add_argument("--activation", choices=("tanh", "relu"), default="tanh")
    ap.add_argument("--sigma-b2", type=float, default=1.0)

    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--classes", nargs=2, type=int, default=list(data_mod.DEFAULT_CLASSES))
    ap.add_argument("--hw", type=int, default=16)
    ap.add_argument("--data-seed", type=int, default=0)

    ap.add_argument("--c-lr", type=float, default=1.0, help="eta = c / lambda_max(Theta_0(X, X))")
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--target-mse", type=float, default=0.1,
                    help="stop at the first step with training MSE <= this")
    ap.add_argument("--kernel-probe", type=int, default=256,
                    help="Theta_t is measured on the first P training points")
    ap.add_argument("--float32", action="store_true",
                    help="train in float32 (TF32 disabled); default float64")

    ap.add_argument("--device", default=None)
    ap.add_argument("--row-block", type=int, default=None,
                    help="Jacobian rows at once; 0 = choose automatically")
    ap.add_argument("--chunk-size", type=int, default=None, help="vmap chunk size in jacrev")
    ap.add_argument("--param-block", type=int, default=None,
                    help="accumulate the Gram matrix over parameter groups of <= N entries")
    ap.add_argument("--jac-device", default=None, help="keep the Jacobian on this device, e.g. cpu")
    ap.add_argument("--grad-chunk", type=int, default=None,
                    help="accumulate the full-batch gradient over chunks of N points")
    ap.add_argument("--partial-every-min", type=float, default=0,
                    help="write a resumable mid-run snapshot every N minutes (0 = off)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.float32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.set_default_dtype(torch.float64)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cells = [RunConfig(arch=args.arch, width=w, depth=L, kernel_size=args.kernel_size,
                       sigma_w2=s, sigma_b2=args.sigma_b2, activation=args.activation,
                       readout=args.readout, padding_mode=args.padding_mode, hw=args.hw,
                       n_train=args.n_train, n_test=args.n_test, classes=tuple(args.classes),
                       data_seed=args.data_seed, c_lr=args.c_lr, max_steps=args.max_steps,
                       target_mse=args.target_mse, kernel_probe=args.kernel_probe)
             for s, w, L in itertools.product(args.sigma_w2, args.widths, args.depths)]
    run_dir = os.path.join(args.out_dir, "runs")
    os.makedirs(run_dir, exist_ok=True)

    ds = data_mod.build_dataset(hw=args.hw, n_train=args.n_train, n_test=args.n_test,
                                classes=tuple(args.classes), seed=args.data_seed,
                                input_scale="sqrt_dim")
    fingerprint = data_mod.data_fingerprint(ds)
    print(f"out={args.out_dir}  device={device}  dtype={torch.get_default_dtype()}  "
          f"runs={len(cells) * len(args.seeds)}  data {ds['meta']['slug']} ({fingerprint})")
    with open(os.path.join(args.out_dir, "sweep_meta.json"), "w") as fh:
        json.dump({"argv": sys.argv[1:], "torch": torch.__version__, "device": str(device),
                   "dtype": str(torch.get_default_dtype()), "data_fingerprint": fingerprint,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)

    t0, n_new = time.time(), 0
    for cfg in cells:
        row_block = suggest_row_block(cfg, args.n_train) if args.row_block == 0 else args.row_block
        for seed in args.seeds:
            name = f"{cfg.slug()}_seed{seed}.pt"
            path = os.path.join(run_dir, name)
            if os.path.exists(path):
                continue
            print(f"{cfg.slug()} seed={seed} P={n_params(build_model(cfg)):,}")
            partial = os.path.join(args.out_dir, "partial", name)
            record = run_training(
                cfg, seed, ds, device=device, row_block=row_block, chunk_size=args.chunk_size,
                jac_device=args.jac_device, param_block=args.param_block,
                grad_chunk=args.grad_chunk, partial_path=partial,
                partial_every_s=60 * args.partial_every_min, verbose=not args.quiet)
            record["data_fingerprint"] = fingerprint
            record["dtype"] = str(torch.get_default_dtype())
            tmp = f"{path}.tmp.{os.getpid()}"      # sweeps over overlapping grids may race
            torch.save(record, tmp)
            os.replace(tmp, path)
            if os.path.exists(partial):
                os.remove(partial)
            n_new += 1
    print(f"done: {n_new} new runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
