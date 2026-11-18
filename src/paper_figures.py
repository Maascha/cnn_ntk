"""The paper's figures, as PDF (TrueType fonts), one file per sigma_w^2 panel.

  fig1_init_LoverC_kernel_imgavg_shade       Fig. 1  D_0 vs L/C, k = 3, 5, 7
  fig2_train_rot_lambda                      Fig. 2  final rotation R_T vs L/C (CNN, k = 3)
  fig3_init_vs_rot_sharex_nolines            Fig. 3  R_T vs D_0
  fig1_init_LoverC_kernel_imgavg_mlp_red     Fig. 4  D_0, CNN k = 3 and MLP (M = 9C)
  fig2_train_rot_lambda_mlp_red              Fig. 5  R_T, CNN and MLP
  fig2_train_delta_lambda_mlp_red            Fig. 6  Delta_T, CNN and MLP
  fig3_init_vs_rot_withmlp_sharex_nolines    Fig. 7  R_T vs D_0, CNN and MLP
  fig4_test_acc_lambda_mlp_red               Fig. 8  test accuracy
  figA3_steps_lambda_mlp_red                 Fig. 9  steps to training MSE 0.1

Inputs:
  <caches>/cnn_k{3,5,7}_img<j>.pt, <caches>/mlp_img<j>.pt   run_dispersion.py
  <cnn-results>/runs/*.pt, <mlp-results>/runs/*.pt           run_training.py

    python src/paper_figures.py --caches caches --cnn-results results/cnn_k3 \
        --mlp-results results/mlp --out figures
"""
import argparse
import glob
import os
import re

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,           # TrueType, no Type 3
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 4,
})
SIG = (1.0, 3.44, 5.0)
LEGFS = 7                                            # legend font size
PANEL = (2.8, 1.85)                                  # inches per single panel
MLPRED = "#d62728"                                   # the MLP in the appendix figures
LS = (2, 4, 8, 12, 16, 24, 32)                       # depths of the dispersion grid
NOTRAIN_MSE = 0.99   # "no training": final training MSE above this (it starts at exactly 1). Runs end either
                     # <= 0.2 (trained, possibly at the step cap) or at ~1 (never trained; rotation = float32 round-off)
YLAB_DISP = r"excess dispersion $D_0$"               # D_0(x) = E[Theta_0(x,x)^2]/E[Theta_0(x,x)]^2 - 1
YLAB_ROT = r"rotation $R_T$"                         # R_T = 1 - <Theta_T,Theta_0>_F/(|Theta_T|_F |Theta_0|_F)
YLAB_DELTA = r"relative change $\Delta_T$"           # Delta_T = |Theta_T - Theta_0|_F / |Theta_0|_F
OUT = "figures"


def ccol(W):
    """One colour per channel count C, shared by all figures: viridis on a log scale over C = 16..256."""
    return plt.cm.viridis(matplotlib.colors.LogNorm(16, 256)(W) * 0.92)


def stag(s):
    return f"s{s:g}".replace(".", "p")


# ------------------------------------------------------------------------------------------------ inputs
def load_grid(path, width_div=1):
    """{(sigma, C, L): jackknife ratio} of one run_dispersion.py output. width_div = 9 keys an MLP
    of width M by the matched channel count C = M / 9."""
    c = torch.load(path, weights_only=False, map_location="cpu")
    return {(float(s), int(W) // width_div, int(L)): float(torch.as_tensor(c["jackknife_ratio_grids"][pi])[si, li])
            for pi, W in enumerate(c["panel_values"]) for si, s in enumerate(c["sigma_w2_values"])
            for li, L in enumerate(c["depths"])}


def load_images(caches, prefix, width_div=1):
    """{image index: grid} for every <caches>/<prefix>_img<j>.pt."""
    out = {}
    for f in glob.glob(os.path.join(caches, f"{prefix}_img*.pt")):
        m = re.fullmatch(rf"{re.escape(prefix)}_img(\d+)\.pt", os.path.basename(f))
        if m:
            out[int(m.group(1))] = load_grid(f, width_div)
    return out


def run_summaries(res, out_dir, seeds):
    """One record per run file in <res>/runs (only what the figures use), cached in
    <out_dir>/summary_<name of res>.pt and rebuilt when a run file is newer."""
    runs = glob.glob(os.path.join(res, "runs", "*.pt"))
    if not runs:
        raise SystemExit(f"no run files in {res}/runs")
    cache = os.path.join(out_dir, f"summary_{os.path.basename(os.path.normpath(res))}.pt")
    if os.path.exists(cache) and os.path.getmtime(cache) >= max(os.path.getmtime(f) for f in runs):
        recs = torch.load(cache, weights_only=False)
    else:
        recs = []
        for f in runs:
            d = torch.load(f, map_location="cpu", weights_only=False)
            c, fin = d["config"], d["checkpoints"][-1] if d["checkpoints"] else {}
            recs.append(dict(arch=c["arch"], sigma=float(c["sigma_w2"]), L=int(c["depth"]), W=int(c["width"]),
                             seed=int(d["seed"]), diverged=bool(d["diverged"]), t01=d["matched_step"],
                             max_steps=int(c["max_steps"]), last_mse=float(d["train_mse_curve"][-1]),
                             rot=fin.get("d"), delta=fin.get("delta"), test_acc=fin.get("test_acc")))
        torch.save(recs, cache)
    return [r for r in recs if r["seed"] in seeds]


def train_data(recs, width_div=1):
    """Trained cells {(sigma, L, C): per-seed lists} (final state), at least 3 seeds, C <= 256. width_div = 9
    keys an MLP of width M by C = M / 9, so it plots at lambda = L/C = 9L/M next to the k = 3 CNN.
    Cells in which no seed trained are returned separately (marked x / + in the figures)."""
    data = {}
    for r in recs:
        v = data.setdefault((r["sigma"], r["L"], r["W"] // width_div), {})
        v.setdefault("n_seen", []).append(1)
        if r["diverged"]:
            continue
        if r["t01"] is None:
            v.setdefault("n_cap", []).append(1)
        v.setdefault("train_mse", []).append(r["last_mse"])      # final training MSE per seed
        for q in ("rot", "delta", "test_acc"):
            if r[q] is not None:
                v.setdefault(q, []).append(float(r[q]))
    data = {k: v for k, v in data.items() if len(v["n_seen"]) >= 3 and k[2] <= 256}
    mse = lambda v: v.get("train_mse") or []                                              # noqa: E731
    notrain = {k for k, v in data.items() if mse(v) and min(mse(v)) > NOTRAIN_MSE}
    print(f"  cells without any training (marked x / +): {sorted(notrain)}")
    # A cell where only SOME seeds never trained: drop those seeds from the cell (else their round-off
    # rotation stretches the min-max band down to 0).
    for k, v in data.items():
        r = mse(v)
        bad = [i for i, x in enumerate(r) if x > NOTRAIN_MSE]
        if bad and k not in notrain:
            for q, vals in v.items():
                if isinstance(vals, list) and len(vals) == len(r):
                    v[q] = [x for i, x in enumerate(vals) if i not in bad]
            print(f"  cell {k}: {len(bad)} of {len(r)} seeds never trained, dropped from the cell")
    return {k: v for k, v in data.items() if k not in notrain}, notrain


# ------------------------------------------------------------------------------------------------ layout
def emit(name, draw, legend=None, cbar=None, ylabel="", xlabel="", sharex=True, leg_panel=0, leg_loc="lower right",
         leg_ncol=2, sig_loc="title", leg_outside=False, sig_y=0.04, ypad_bottom=1.0):
    """draw(ax, sigma) draws one panel. legend: list of handles. cbar: (norm, cmap, label).
    Writes the row of three panels, every single panel (same limits), and the legend / colourbar."""
    d = OUT
    os.makedirs(d, exist_ok=True)
    legkw = dict(loc=leg_loc, ncol=leg_ncol, frameon=True, framealpha=0.9, edgecolor="none", fontsize=LEGFS,
                 handlelength=1.6, columnspacing=0.8, borderpad=0.3, labelspacing=0.25, handletextpad=0.4)

    def siglabel(a, s):
        if sig_loc == "title":
            a.set_title(rf"$\sigma_w^2={s:g}$", fontsize=9, pad=3)
        else:                                          # inside the axes, in an empty corner
            x, ha = (0.97, "right") if "right" in sig_loc else (0.03, "left")
            y, va = (sig_y, "bottom") if "lower" in sig_loc else (0.96, "top")
            a.text(x, y, rf"$\sigma_w^2={s:g}$", transform=a.transAxes, ha=ha, va=va, fontsize=9, zorder=5,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    fig, axs = plt.subplots(1, 3, figsize=(3 * PANEL[0], PANEL[1] + 0.15), sharey=True, constrained_layout=True)
    for a, s in zip(axs, SIG):
        draw(a, s)
        siglabel(a, s)
        a.set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    if sharex:
        x0 = min(a.get_xlim()[0] for a in axs)
        x1 = max(a.get_xlim()[1] for a in axs)
        for a in axs:
            a.set_xlim(x0, x1)
    if ypad_bottom != 1.0:                             # extra room below the data (log axis)
        y0, y1 = axs[0].get_ylim()
        axs[0].set_ylim(y0 / ypad_bottom, y1)
    xl = {s: a.get_xlim() for s, a in zip(SIG, axs)}
    yl = axs[0].get_ylim()
    if legend and leg_outside:
        fig.legend(handles=legend, loc="outside lower center", ncol=(len(legend) + 1) // 2, frameon=False,
                   fontsize=LEGFS, handlelength=1.6, columnspacing=1.0)
        fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] + 0.4)
    elif legend:
        axs[leg_panel].legend(handles=legend, **legkw)
    if cbar:
        fig.colorbar(plt.cm.ScalarMappable(norm=cbar[0], cmap=cbar[1]), ax=list(axs), label=cbar[2], pad=0.01)
    fig.savefig(os.path.join(d, f"{name}_all.pdf"))
    plt.close(fig)
    for s in SIG:                                      # single panels, same limits as in the row
        for with_leg in (False, True):
            fig, a = plt.subplots(figsize=(PANEL[0], PANEL[1] + 0.15), constrained_layout=True)
            draw(a, s)
            a.set_xlim(*xl[s])
            a.set_ylim(*yl)
            siglabel(a, s)
            a.set_xlabel(xlabel)
            a.set_ylabel(ylabel)
            if with_leg and cbar:
                fig.colorbar(plt.cm.ScalarMappable(norm=cbar[0], cmap=cbar[1]), ax=a, label=cbar[2], pad=0.01)
                fig.set_size_inches(PANEL[0] + 0.75, PANEL[1] + 0.15)
            if with_leg and legend and leg_outside:
                fig.legend(handles=legend, loc="outside right center", ncol=1, frameon=False, fontsize=LEGFS,
                           handlelength=1.6)
                fig.set_size_inches(PANEL[0] + 1.05 + (0.75 if cbar else 0), PANEL[1] + 0.15)
            elif with_leg and legend:
                a.legend(handles=legend, **legkw)
            fig.savefig(os.path.join(d, f"{name}_{stag(s)}{'_leg' if with_leg else ''}.pdf"))
            plt.close(fig)
    if legend:
        fig = plt.figure(figsize=(1.6, 1.2))
        fig.legend(handles=legend, loc="center", ncol=1 if len(legend) <= 8 else 2, frameon=False, fontsize=LEGFS)
        fig.savefig(os.path.join(d, f"{name}_legend.pdf"), bbox_inches="tight")
        plt.close(fig)
        fig = plt.figure(figsize=(3 * PANEL[0], 0.4))
        fig.legend(handles=legend, loc="center", ncol=len(legend), frameon=False, handlelength=1.5,
                   columnspacing=1.0, fontsize=LEGFS)
        fig.savefig(os.path.join(d, f"{name}_legend_row.pdf"), bbox_inches="tight")
        plt.close(fig)
        fig = plt.figure(figsize=(3 * PANEL[0], 0.6))      # the same entries in two rows, widths in row 1
        nw = sum(1 for h in legend if h.get_label().startswith(("$C=", "$M=")))
        nc = max(nw, (len(legend) + 1) // 2)
        order = [legend[i] for c in range(nc) for i in (c, c + nc) if i < len(legend)]   # legend fills column-wise
        fig.legend(handles=order, loc="center", ncol=nc, frameon=False, handlelength=1.5, columnspacing=1.0,
                   fontsize=LEGFS)
        fig.savefig(os.path.join(d, f"{name}_legend_2rows.pdf"), bbox_inches="tight")
        plt.close(fig)
    if cbar:
        fig, a = plt.subplots(figsize=(0.75, PANEL[1]), constrained_layout=True)
        fig.colorbar(plt.cm.ScalarMappable(norm=cbar[0], cmap=cbar[1]), cax=a, label=cbar[2])
        fig.savefig(os.path.join(d, f"{name}_cbar.pdf"))
        plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------ figures
def fig1(per, with_mlp=False):
    """Image average (mean +- standard error) of D_0 vs lambda = L/C, one curve per (C, k).
    with_mlp=False: k = 3, 5, 7, hue = C, shade / line style / marker = k (Fig. 1).
    with_mlp=True: k = 3 and the MLP at M = 9C as a red line, marker filled in the C colour (Fig. 4).
    Every curve of a figure averages over the same images (those all its architectures have)."""
    Ws = [16, 32, 64, 128, 256]
    ks = (3, "mlp") if with_mlp else (3, 5, 7)
    ids = sorted(set.intersection(*(set(per[k]) for k in ks)))
    imgs = {k: [per[k][i] for i in ids] for k in ks}
    colC = {W: ccol(W) for W in Ws}
    mkK = {3: "o", 5: "s", 7: "^"}
    LSK = {3: "-", 5: "--", 7: ":"}

    def shade(c, k):                                   # k3 darker, k5 the C colour, k7 lighter
        c = np.array(matplotlib.colors.to_rgb(c))
        return {3: tuple(c * 0.6), 5: tuple(c), 7: tuple(c + (1 - c) * 0.45)}[k]

    def series(k, s, W):
        Ls, m, se = [], [], []
        for L in LS:
            v = [d[(s, W, L)] - 1 for d in imgs[k] if (s, W, L) in d]
            if len(v) == len(imgs[k]):
                Ls.append(L)
                m.append(np.mean(v))
                se.append(np.std(v, ddof=1) / np.sqrt(len(v)))
        return Ls, m, se

    def draw(a, s):
        for k in ks:
            for W in Ws:
                Ls, y, se = series(k, s, W)
                if not Ls:
                    continue
                x = [L / W for L in Ls]
                if k == "mlp":
                    a.plot(x, y, ls="-", marker="o", ms=3.4, lw=1.0, color=MLPRED, mfc=colC[W], mec=MLPRED, mew=0.9,
                           alpha=.9)
                    c = MLPRED
                elif with_mlp:
                    c = colC[W]
                    a.plot(x, y, ls="-", marker="o", ms=3.2, lw=1.0, color=c, alpha=.9)
                else:
                    c = shade(colC[W], k)
                    a.plot(x, y, ls=LSK[k], marker=mkK[k], ms=3.2, lw=1.0, color=c, alpha=.9)
                a.errorbar(x, y, yerr=se, fmt="none", ecolor=c, lw=0.7, alpha=.8, capsize=0)
        a.set_xscale("log", base=2)
        a.set_yscale("log", base=2)
        a.grid(alpha=.25, lw=0.4)
    leg = [Line2D([], [], color=c, lw=1.5, label=f"$C={W}$") for W, c in colC.items()]
    if with_mlp:
        leg += [Line2D([], [], color="0.3", marker="o", ls="-", ms=4, lw=1.0, label="CNN, $k=3$"),
                Line2D([], [], color=MLPRED, marker="o", ls="-", ms=4, lw=1.0, label="MLP, $M=9C$",
                       mfc="0.6", mec=MLPRED, mew=0.9)]
    else:
        leg += [Line2D([], [], color=shade((0.35, 0.35, 0.35), k), marker=mkK[k], ls=LSK[k], ms=4, lw=1.0,
                       label=f"$k={k}$") for k in ks]
    name = "fig1_init_LoverC_kernel_imgavg" + ("_mlp_red" if with_mlp else "_shade")
    print(f"  {name}: images {ids} (n = {len(ids)})")
    emit(name, draw, legend=leg, ylabel=YLAB_DISP, xlabel=r"$\lambda=L/C$", leg_panel=1, leg_loc="upper left",
         sig_loc="lower right")


def fig2(data, notrain, mlp=None, qs=("rot",)):
    """Final rotation R_T (and relative change Delta_T) vs lambda = L/C, one line per C: median over seeds,
    seed min-max band; open marker = a seed hit the step cap, x = no seed trained. mlp = (data, notrain)
    of the MLP keyed by C = M/9, drawn as a red line with markers filled in the C colour (suffix _mlp_red)."""
    Ws = sorted({k[2] for k in data})
    col = {W: ccol(W) for W in Ws}
    for q, ylab in (("rot", YLAB_ROT), ("delta", YLAB_DELTA)):
        if q not in qs:
            continue

        def draw(a, s, q=q):
            if mlp:
                for W in Ws:
                    pts = sorted((L / W, np.median(v[q])) for (ss, L, w), v in mlp[0].items()
                                 if ss == s and w == W and v.get(q))
                    if len(pts) > 1:
                        a.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=MLPRED, lw=0.9, zorder=2,
                               marker="o", ms=3.2, mfc=col[W], mec=MLPRED, mew=0.9)
            for W in Ws:
                pts = sorted((L / W, v[q], len(v.get("n_cap", [])) > 0) for (ss, L, w), v in data.items()
                             if ss == s and w == W and v.get(q))
                if len(pts) < 2:
                    continue
                x = np.array([p[0] for p in pts])
                y = np.array([np.median(p[1]) for p in pts])
                lo = np.array([np.min(p[1]) for p in pts])
                hi = np.array([np.max(p[1]) for p in pts])
                cap = np.array([p[2] for p in pts])
                a.plot(x, y, "-", color=col[W], lw=1.0, zorder=2)
                a.fill_between(x, lo, hi, color=col[W], alpha=0.15, lw=0)
                a.scatter(x[~cap], y[~cap], s=9, color=col[W], zorder=3, lw=0)
                a.scatter(x[cap], y[cap], s=14, facecolor="white", edgecolor=col[W], zorder=3, lw=0.9)
            tr = matplotlib.transforms.blended_transform_factory(a.transData, a.transAxes)
            nt = [(L / W, W) for (ss, L, W) in notrain if ss == s]
            if nt:
                a.scatter([x for x, _ in nt], [0.04] * len(nt), marker="x", s=22, lw=1.1, color=[col[W] for _, W in nt],
                          transform=tr, zorder=5, clip_on=False)
            if mlp:
                ntm = [(L / W, W) for (ss, L, W) in mlp[1] if ss == s and W in col]
                if ntm:
                    a.scatter([x for x, _ in ntm], [0.04] * len(ntm), marker="+", s=26, lw=1.1, color=MLPRED,
                              transform=tr, zorder=4, clip_on=False)
            a.set_xscale("log")
            a.set_yscale("log")
            a.grid(alpha=.25, lw=0.4, which="major")
        leg = [Line2D([], [], color=c, lw=1.5, marker="o", ms=3, label=f"$C={W}$") for W, c in col.items()] + \
              [Line2D([], [], color="0.3", marker="o", ms=4, ls="", mfc="white", label="step cap hit"),
               Line2D([], [], color="0.3", marker="x", ms=4, ls="", label="no training, CNN" if mlp else "no training")]
        if mlp:
            leg += [Line2D([], [], color=MLPRED, marker="+", ms=5, ls="", label="no training, MLP"),
                    Line2D([], [], color="0.3", lw=1.5, label="CNN $k=3$"),
                    Line2D([], [], color=MLPRED, lw=0.9, ls="-", marker="o", ms=3, mfc="0.6", mec=MLPRED,
                           label="MLP $M=9C$")]
        emit(f"fig2_train_{q}_lambda" + ("_mlp_red" if mlp else ""), draw, legend=leg, ylabel=ylab,
             xlabel=r"$\lambda=L/C$", leg_outside=True, sig_loc="lower left", ypad_bottom=2.0)


def fig3(data, imgs_cnn, mlp=None, imgs_mlp=None):
    """Final rotation (median over seeds) vs initial excess dispersion (mean over images), error bars = min-max
    ranges, colour = lambda, marker = C. mlp: + the MLP (keyed by C = M/9) with red marker edges (_withmlp)."""
    MK = {16: "o", 24: "<", 32: "s", 48: "v", 64: "^", 96: ">", 128: "D", 256: "P"}

    def points(data, imgs):
        pts = {}
        for (s, L, W), v in data.items():
            if W not in MK or not v.get("rot"):
                continue
            x = [d[(s, W, L)] - 1 for d in imgs if (s, W, L) in d]
            if len(x) < len(imgs) or np.mean(x) <= 0:
                continue
            pts[(s, W, L)] = (np.mean(x), float(np.median(v["rot"])), len(v.get("n_cap", [])) > 0, np.min(x),
                              np.max(x), float(np.min(v["rot"])), float(np.max(v["rot"])))
        return pts
    sets = {"cnn": (points(data, imgs_cnn), len(imgs_cnn))}
    if mlp is not None:
        sets["mlp"] = (points(mlp, imgs_mlp), len(imgs_mlp))
    lams = [k[2] / k[1] for pts, _ in sets.values() for k in pts]
    norm = matplotlib.colors.LogNorm(min(lams), max(lams))
    cmap = plt.cm.cividis

    def draw(a, s):
        for arch, (pts, _) in sets.items():
            edge = MLPRED if arch == "mlp" else "k"
            for k, p in pts.items():
                if k[0] != s:
                    continue
                c = cmap(norm(k[2] / k[1]))
                a.errorbar(p[0], p[1], xerr=[[p[0] - p[3]], [p[4] - p[0]]], yerr=[[p[1] - p[5]], [p[6] - p[1]]],
                           fmt="none", ecolor=c, alpha=.7, lw=0.8, capsize=1.5, zorder=2)
                a.scatter(p[0], p[1], marker=MK[k[1]], s=26, facecolor="white" if p[2] else c,
                          edgecolor=c if p[2] else edge, lw=1.0 if p[2] or edge != "k" else 0.5, zorder=3)
        a.set_xscale("log")
        a.set_yscale("log")
        a.grid(alpha=.25, lw=0.4, which="major")
    leg = [Line2D([], [], marker=m, ls="none", mfc="0.6", mec="k", mew=0.5, ms=4, label=f"$C={W}$") for W, m in MK.items()] + \
          [Line2D([], [], marker="o", ls="none", mfc="white", mec="0.3", mew=0.9, ms=4, label="step cap hit")]
    if mlp is not None:
        leg += [Line2D([], [], marker="o", ls="none", mfc="0.6", mec="k", mew=0.5, ms=4, label="CNN, $k=3$"),
                Line2D([], [], marker="o", ls="none", mfc="0.6", mec=MLPRED, mew=1.0, ms=4, label="MLP, $M=9C$")]
    name = "fig3_init_vs_rot" + ("_withmlp" if mlp is not None else "") + "_sharex_nolines"
    emit(name, draw, legend=leg, cbar=(norm, cmap, r"$\lambda=L/C$"), ylabel=YLAB_ROT, xlabel=YLAB_DISP,
         sharex=True, leg_outside=True, sig_loc="lower right")
    for arch, (pts, nimg) in sets.items():
        for s in SIG:
            sub = [p for k, p in pts.items() if k[0] == s]
            if len(sub) > 2:
                print(f"  {name} {arch} sigma {s:g}: n = {len(sub)}, Spearman rho = "
                      f"{spearmanr([p[0] for p in sub], [p[1] for p in sub])[0]:.2f}, images = {nimg}")


def fig4(data, notrain, mlp):
    """Test accuracy at the end of training vs lambda, one line per C (median over seeds, min-max band). As in fig2:
    a cell where a seed hit the step cap has an open marker (accuracy at the cap); cells without training are not
    drawn and marked x (CNN) / + (MLP) at the bottom. mlp = (data, notrain) of the MLP (C = M/9), as red lines."""
    Ws = sorted({k[2] for k in data})
    col = {W: ccol(W) for W in Ws}

    def line(a, d, W, s, is_mlp):
        pts = sorted((L / W, v["test_acc"], len(v.get("n_cap", [])) > 0) for (ss, L, w), v in d.items()
                     if ss == s and w == W and v.get("test_acc"))
        if len(pts) < 2:
            return
        x = np.array([p[0] for p in pts])
        y = np.array([np.median(p[1]) for p in pts])
        cp = np.array([p[2] for p in pts])
        if is_mlp:
            a.plot(x, y, "-", color=MLPRED, lw=0.9, marker="o", ms=3.2, mfc=col[W], mec=MLPRED, mew=0.9, zorder=2)
        else:
            a.plot(x, y, "-", color=col[W], lw=1.0, zorder=3)
            a.fill_between(x, [np.min(p[1]) for p in pts], [np.max(p[1]) for p in pts], color=col[W], alpha=0.15, lw=0)
            a.scatter(x[~cp], y[~cp], s=9, color=col[W], zorder=4, lw=0)
            a.scatter(x[cp], y[cp], s=14, facecolor="white", edgecolor=col[W], zorder=4, lw=0.9)

    def draw(a, s):
        for W in Ws:
            line(a, data, W, s, False)
            line(a, mlp[0], W, s, True)
        tr = matplotlib.transforms.blended_transform_factory(a.transData, a.transAxes)
        nt = [(L / W, W) for (ss, L, W) in notrain if ss == s and W in col]
        if nt:                                                    # CNN x above the MLP +
            a.scatter([x for x, _ in nt], [0.04] * len(nt), marker="x", s=22, lw=1.1, color=[col[W] for _, W in nt],
                      transform=tr, zorder=5, clip_on=False)
        ntm = [(L / W, W) for (ss, L, W) in mlp[1] if ss == s and W in col]
        if ntm:
            a.scatter([x for x, _ in ntm], [0.04] * len(ntm), marker="+", s=26, lw=1.1, color=MLPRED,
                      transform=tr, zorder=4, clip_on=False)
        a.set_xscale("log")
        a.grid(alpha=.25, lw=0.4, which="major")
    leg = [Line2D([], [], color=c, lw=1.5, marker="o", ms=3, label=f"$C={W}$") for W, c in col.items()] + \
          [Line2D([], [], color="0.3", marker="o", ms=4, ls="", mfc="white", label="step cap hit"),
           Line2D([], [], color="0.3", marker="x", ms=4, ls="", label="no training, CNN"),
           Line2D([], [], color=MLPRED, marker="+", ms=5, ls="", label="no training, MLP"),
           Line2D([], [], color="0.3", lw=1.5, label="CNN $k=3$"),
           Line2D([], [], color=MLPRED, lw=0.9, ls="-", marker="o", ms=3, mfc="0.6", mec=MLPRED, label="MLP $M=9C$")]
    emit("fig4_test_acc_lambda_mlp_red", draw, legend=leg, ylabel="test accuracy", xlabel=r"$\lambda=L/C$",
         leg_outside=True, sig_loc="lower left")


def figA3(recs, mlp_recs):
    """Steps to training MSE 0.1 vs lambda, one line per C: median over seeds, seed min-max band; a seed at the step
    cap counts as the cap (open marker), cells where no seed trained (final training MSE > NOTRAIN_MSE, as train_data) are marked
    x (CNN) / + (MLP). MLP keyed by C = M/9, as red lines."""
    def cells_of(rs, width_div=1):
        out = {}
        for r in rs:
            out.setdefault((r["sigma"], r["L"], r["W"] // width_div), []).append(r)
        return {k: v for k, v in out.items() if len({r["seed"] for r in v}) >= 3 and k[2] <= 256}
    cells, mlp_cells = cells_of(recs), cells_of(mlp_recs, 9)
    stuck = lambda r: r["last_mse"] > NOTRAIN_MSE                                                  # noqa: E731
    Ws = sorted({k[2] for k in cells})
    col = {W: ccol(W) for W in Ws}
    cap = max(r["max_steps"] for r in recs)

    def series(cs, s, W):
        pts = []
        for (ss, L, w), rs in sorted(cs.items(), key=lambda kv: kv[0][1]):
            if ss != s or w != W:
                continue
            t = [r["t01"] if r["t01"] is not None else cap for r in rs]
            pts.append((L / W, np.median(t), min(t), max(t), any(r["t01"] is None for r in rs), all(stuck(r) for r in rs)))
        return [p for p in pts if not p[5]]

    def draw(a, s):
        for W in Ws:
            pts = series(cells, s, W)
            if pts:
                x = np.array([p[0] for p in pts])
                y = np.array([p[1] for p in pts])
                cp = np.array([p[4] for p in pts])
                a.plot(x, y, "-", color=col[W], lw=1.0, zorder=2)
                a.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=col[W], alpha=0.15, lw=0)
                a.scatter(x[~cp], y[~cp], s=9, color=col[W], zorder=3, lw=0)
                a.scatter(x[cp], y[cp], s=14, facecolor="white", edgecolor=col[W], zorder=3, lw=0.9)
            mp = series(mlp_cells, s, W)
            if mp:
                a.plot([p[0] for p in mp], [p[1] for p in mp], "-", color=MLPRED, lw=0.9, marker="o", ms=3.2,
                       mfc=col[W], mec=MLPRED, mew=0.9, zorder=2)
        for cs, red in ((cells, False), (mlp_cells, True)):
            nt = [(L / W, W) for (ss, L, W), rs in cs.items() if ss == s and all(stuck(r) for r in rs)]
            if nt:
                a.scatter([x for x, _ in nt], [cap] * len(nt), marker="+" if red else "x", s=26 if red else 22, lw=1.1,
                          color=MLPRED if red else [col[W] for _, W in nt], zorder=4 if red else 5, clip_on=False)
        a.axhline(cap, color="0.5", lw=0.6, ls=":", zorder=1)
        a.set_xscale("log")
        a.set_yscale("log")
        a.grid(alpha=.25, lw=0.4, which="major")
    leg = [Line2D([], [], color=c, lw=1.5, marker="o", ms=3, label=f"$C={W}$") for W, c in col.items()] + \
          [Line2D([], [], color="0.3", marker="o", ms=4, ls="", mfc="white", label="a seed at the step cap"),
           Line2D([], [], color="0.3", marker="x", ms=4, ls="", label="no training, CNN"),
           Line2D([], [], color=MLPRED, marker="+", ms=5, ls="", label="no training, MLP"),
           Line2D([], [], color="0.5", lw=0.6, ls=":", label="step cap ($10^5$)" if cap == 100_000 else f"step cap ({cap})"),
           Line2D([], [], color="0.3", lw=1.5, label="CNN $k=3$"),
           Line2D([], [], color=MLPRED, lw=0.9, marker="o", ms=3, mfc="0.6", mec=MLPRED, label="MLP $M=9C$")]
    emit("figA3_steps_lambda_mlp_red", draw, legend=leg, ylabel="steps to training MSE 0.1", xlabel=r"$\lambda=L/C$",
         leg_outside=True, sig_loc="lower left")
    for name, cs in (("CNN", cells), ("MLP", mlp_cells)):
        for s in SIG:
            ks = [k for k in cs if k[0] == s]
            nc = sum(any(r["t01"] is None for r in cs[k]) for k in ks)
            nn = sum(all(stuck(r) for r in cs[k]) for k in ks)
            print(f"  figA3 {name} sigma {s:g}: {len(ks)} cells, {nc} with a seed at the cap, {nn} with no seed trained")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", default="caches", help="directory with the run_dispersion.py outputs")
    ap.add_argument("--cnn-results", default="results/cnn_k3", help="run_training.py --out-dir of the k = 3 CNN")
    ap.add_argument("--mlp-results", default="results/mlp", help="run_training.py --out-dir of the MLP (M = 9C)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="+", choices=("init", "train"), default=("init", "train"),
                    help="init: Figs. 1, 4 (dispersion caches only); train: all others")
    args = ap.parse_args()
    OUT = args.out

    per = {k: load_images(args.caches, f"cnn_k{k}") for k in (3, 5, 7)}
    per["mlp"] = load_images(args.caches, "mlp", width_div=9)
    if "init" in args.only:
        fig1(per)
        fig1(per, with_mlp=True)
    if "train" in args.only:
        os.makedirs(OUT, exist_ok=True)
        recs = run_summaries(args.cnn_results, OUT, set(args.seeds))
        mrecs = run_summaries(args.mlp_results, OUT, set(args.seeds))
        print("CNN:")
        data, notrain = train_data(recs)
        print("MLP:")
        mlp = train_data(mrecs, width_div=9)
        fig2(data, notrain)
        fig2(data, notrain, mlp, qs=("rot", "delta"))
        # the images on the full grid: those with an MLP cache too 
        ids = sorted(set(per[3]) & set(per["mlp"]))
        imgs_cnn = [per[3][i] for i in ids]
        imgs_mlp = [per["mlp"][i] for i in ids]
        print(f"  fig3: images {ids} (n = {len(ids)})")
        fig3(data, imgs_cnn)
        fig3(data, imgs_cnn, mlp[0], imgs_mlp)
        fig4(data, notrain, mlp)
        figA3(recs, mrecs)


if __name__ == "__main__":
    main()
