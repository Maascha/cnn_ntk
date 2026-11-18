"""One training run (Figs. 2, 3, 5-9).

Centered network f(x; theta) - f(x; theta_0), summed half-squared loss
0.5 * sum_i (f(x_i) - y_i)^2 over the full training set, plain gradient descent
(no momentum, no weight decay), learning rate eta = c_lr / lambda_max(K_0) with
K_0 = Theta_0(X, X) on the whole training set. With this loss the GD step on the
linearized model is r <- (I - eta K_0) r. The reported training MSE is
mean_i (f(x_i) - y_i)^2 = 2 * loss / n.

Training stops at the first step with training MSE <= cfg.target_mse, or after
cfg.max_steps. At the checkpoints below, Theta_t = Theta(X_P, X_P) on the probe set
(the first cfg.kernel_probe training points) is recomputed and compared with
Theta_0 (kernels.kernel_change: rotation R_t = "d", relative change Delta_t =
"delta"), and train/test MSE and accuracy are recorded.
"""

import math
import os
import time

import torch

import kernels
from models import init_model, n_params, prepare_inputs

DIVERGENCE_MSE = 1e6
DENSE_UNTIL = 2000          # checkpoints every 10 steps up to here,
PER_DECADE = 20             # then 20 per decade


def checkpoint_steps(max_steps):
    steps = {0, max_steps}
    steps.update(s for s in (1, 2, 4, 8) if s <= max_steps)
    steps.update(range(10, min(DENSE_UNTIL, max_steps) + 1, 10))
    s = float(DENSE_UNTIL)
    while s < max_steps:
        s *= 10.0 ** (1.0 / PER_DECADE)
        steps.add(min(int(round(s)), max_steps))
    return steps


def _accuracy(pred, y):
    sign = torch.sign(pred)
    sign[sign == 0] = 1.0
    return float((sign == y).to(torch.float64).mean())


def run_training(cfg, seed, data, device="cpu", *, row_block=None, chunk_size=None,
                 jac_device=None, param_block=None, grad_chunk=None,
                 partial_path=None, partial_every_s=0, verbose=True):
    """Returns the run record (a dict)."""
    t_start = time.time()
    dtype = torch.get_default_dtype()
    X_tr = data["X_train"].to(device=device, dtype=dtype)
    y_tr = data["y_train"].to(device=device, dtype=dtype)
    X_te = data["X_test"].to(device=device, dtype=dtype)
    y_te = data["y_test"].to(device=device, dtype=dtype)
    n = X_tr.shape[0]

    model = init_model(cfg, seed, device=device)
    in_tr, in_te = prepare_inputs(cfg, X_tr), prepare_inputs(cfg, X_te)
    with torch.no_grad():                              # centering offsets f(x; theta_0)
        f0_tr = model(in_tr).squeeze(-1).clone()
        f0_te = model(in_te).squeeze(-1).clone()

    # Learning rate from the full training-set kernel; its P x P corner is Theta_0 on the probe set.
    gram_kw = dict(row_block=row_block, chunk_size=chunk_size,
                   jac_device=jac_device, param_block=param_block)
    K0 = kernels.gram(model, in_tr, **gram_kw)
    K0d = K0.to(torch.float64)
    evals = torch.linalg.eigh(0.5 * (K0d + K0d.T))[0].flip(0)   # descending
    mu1 = float(evals[0])
    if not mu1 > 0:
        raise ValueError(f"Theta_0(X, X) has non-positive top eigenvalue {mu1!r}")
    eta = cfg.c_lr / mu1
    P = cfg.kernel_probe
    in_k = in_tr[:P]
    K0k = K0[:P, :P].clone()
    del K0, K0d

    optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=0.0, weight_decay=0.0)
    planned = checkpoint_steps(cfg.max_steps)
    checkpoints = []
    train_mse_curve = torch.empty(cfg.max_steps + 1, dtype=torch.float64)
    matched_step, diverged, stopped = None, False, False

    # Optional mid-run snapshot, so an interrupted job continues from it. Full-batch GD
    # without momentum has no state besides theta_t, and everything before the loop
    # is a deterministic function of the seed.
    t_begin, wall_before = 0, 0.0
    if partial_path and os.path.exists(partial_path):
        st = torch.load(partial_path, map_location="cpu", weights_only=False)
        if st["slug"] != cfg.slug() or st["seed"] != seed:
            raise ValueError(f"{partial_path} belongs to {st['slug']} seed {st['seed']}")
        model.load_state_dict({k: v.to(device) for k, v in st["params"].items()})
        t_begin, wall_before = st["t"], st["wall_s"]
        train_mse_curve[:t_begin] = st["train_mse_curve"]
        checkpoints, matched_step = st["checkpoints"], st["matched_step"]
        if verbose:
            print(f"    resumed from {partial_path} at step {t_begin}")
    last_save = time.time()

    def save_partial(t_now):
        state = {"slug": cfg.slug(), "seed": seed, "t": t_now,
                 "params": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "train_mse_curve": train_mse_curve[:t_now].clone(),
                 "checkpoints": checkpoints, "matched_step": matched_step,
                 "wall_s": wall_before + time.time() - t_start}
        os.makedirs(os.path.dirname(partial_path), exist_ok=True)
        tmp = f"{partial_path}.tmp.{os.getpid()}"
        torch.save(state, tmp)
        os.replace(tmp, partial_path)

    t = t_begin
    for t in range(t_begin, cfg.max_steps + 1):
        if (partial_path and partial_every_s and t > t_begin
                and time.time() - last_save > partial_every_s):
            save_partial(t)
            last_save = time.time()

        if grad_chunk and grad_chunk < n:
            # Same full-batch gradient, accumulated over slices of the training set (memory only).
            optimizer.zero_grad(set_to_none=True)
            parts, loss_val = [], 0.0
            for lo in range(0, n, grad_chunk):
                hi = min(lo + grad_chunk, n)
                fc_part = model(in_tr[lo:hi]).squeeze(-1) - f0_tr[lo:hi]
                l_part = 0.5 * (fc_part - y_tr[lo:hi]).pow(2).sum()
                l_part.backward()
                parts.append(fc_part.detach())
                loss_val += float(l_part.detach())
            f_c, loss = torch.cat(parts), None
            train_mse = 2.0 * loss_val / n
        else:
            f_c = model(in_tr).squeeze(-1) - f0_tr
            loss = 0.5 * (f_c - y_tr).pow(2).sum()
            train_mse = float(2.0 * loss.detach() / n)
        train_mse_curve[t] = train_mse

        if not math.isfinite(train_mse) or train_mse > DIVERGENCE_MSE:
            diverged = True
            if verbose:
                print(f"    diverged at step {t} (train MSE {train_mse:.3e})")
            break

        stopped = train_mse <= cfg.target_mse
        if stopped:
            matched_step = t
        if t in planned or stopped:
            with torch.no_grad():
                f_te = model(in_te).squeeze(-1) - f0_te
            K_t = kernels.gram(model, in_k, **gram_kw)
            rec = {"step": t, "train_mse": train_mse,
                   "test_mse": float((f_te - y_te).pow(2).mean()),
                   "train_acc": _accuracy(f_c.detach(), y_tr),
                   "test_acc": _accuracy(f_te, y_te)}
            rec.update(kernels.kernel_change(K_t, K0k))
            checkpoints.append(rec)
            if verbose:
                print(f"    step {t:>6}  mse {train_mse:.4e}  Delta {rec['delta']:.4e}  "
                      f"R {rec['d']:.3e}  test acc {rec['test_acc']:.3f}")

        if stopped or t == cfg.max_steps:
            break
        if loss is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        optimizer.step()

    return {
        "config": cfg.to_dict(),
        "seed": int(seed),
        "n_params": n_params(model),
        "eta": float(eta),
        "mu1": mu1,
        "evals": evals.cpu(),                 # spectrum of Theta_0(X, X), descending
        "K0": K0k.detach().cpu(),             # Theta_0 on the probe set
        "checkpoints": checkpoints,           # the last one is the final state T
        "train_mse_curve": train_mse_curve[: t + 1].clone(),
        "matched_step": matched_step,         # first step with training MSE <= target, or None
        "diverged": diverged,
        "steps_run": int(t),
        "wall_time_s": wall_before + time.time() - t_start,
    }
