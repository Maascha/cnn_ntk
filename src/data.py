"""Binary CIFAR-10 data (airplane vs frog, classes 0 and 6).

Preprocessing:
  1. RGB uint8 -> float in [0, 1]
  2. avg_pool2d(32 // hw) to the target resolution (hw=16: 2x2 average pooling)
  3. subtract the per-pixel mean of the training subset (test uses the same mean)
  4. rescale each image to ||x|| = sqrt(C*H*W) ("sqrt_dim", paper) or ||x|| = 1 ("unit")

Labels are +1 / -1. The class-balanced subsets are drawn with a fixed seed and
interleaved (class a, class b, class a, ...), so every even-length prefix of the
training set is balanced as well; the kernel probe set X_P is the first P
training images, and the images used for the dispersion experiments are
training images 0, 1, 2, ...

run `python src/download_cifar.py` once, or set $NTK_CIFAR_ROOT to a directory
containing `cifar-10-batches-py`.
"""

import hashlib
import os

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
DEFAULT_CIFAR_ROOT = os.path.join(ROOT, "data")
DEFAULT_CACHE_DIR = os.path.join(ROOT, "data", "cache")

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")
DEFAULT_CLASSES = (0, 6)          # airplane vs frog
INPUT_SCALES = ("unit", "sqrt_dim")

# The dataset of the paper (training and the dispersion experiments).
PAPER_DATA = dict(hw=16, n_train=2048, n_test=2000, classes=DEFAULT_CLASSES,
                  seed=0, input_scale="sqrt_dim")


def resolve_cifar_root(root=None):
    root = root or os.environ.get("NTK_CIFAR_ROOT") or DEFAULT_CIFAR_ROOT
    if os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
        return root
    raise FileNotFoundError(
        f"no CIFAR-10 copy in {root}. Run `python src/download_cifar.py` once, or "
        "point $NTK_CIFAR_ROOT at a directory containing cifar-10-batches-py.")


def _load_cifar_split(root, train):
    from torchvision import datasets

    ds = datasets.CIFAR10(root=root, train=train, download=False)
    x = torch.from_numpy(ds.data)                  # [N, 32, 32, 3] uint8
    x = x.permute(0, 3, 1, 2).to(torch.get_default_dtype()) / 255.0
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def _balanced_subset(x, y, classes, n, generator):
    """n/2 images of each class without replacement, interleaved; labels +1/-1."""
    if n % 2:
        raise ValueError(f"n must be even for a balanced subset, got {n}")
    picks = []
    for cls in classes:
        idx = (y == cls).nonzero(as_tuple=True)[0]
        perm = torch.randperm(idx.numel(), generator=generator)
        picks.append(idx[perm[: n // 2]])
    order = torch.stack(picks, dim=1).reshape(-1)
    labels = torch.tensor([1.0, -1.0]).repeat(n // 2)
    return x[order], labels, order


def _normalize(x_train, x_test, input_scale):
    if input_scale not in INPUT_SCALES:
        raise ValueError(f"input_scale must be one of {INPUT_SCALES}, got {input_scale!r}")
    mean = x_train.mean(dim=0, keepdim=True)
    target = 1.0 if input_scale == "unit" else float(x_train[0].numel()) ** 0.5
    out = []
    for x in (x_train, x_test):
        x = x - mean
        norms = x.flatten(1).norm(dim=1).clamp_min(1e-12)
        out.append(target * x / norms.view(-1, 1, 1, 1))
    return out


def dataset_slug(hw, n_train, n_test, classes, seed, input_scale):
    cls = "-".join(str(c) for c in classes)
    scale = "" if input_scale == "unit" else f"_{input_scale}"
    return f"cifar_pair_hw{hw}_c3_n{n_train}+{n_test}_cls{cls}{scale}_seed{seed}"


def build_dataset(*, hw=16, n_train=2048, n_test=2000, classes=DEFAULT_CLASSES,
                  seed=0, input_scale="sqrt_dim", root=None, cache_dir=DEFAULT_CACHE_DIR):
    """Dict with X_train [n, 3, hw, hw], y_train [n] in {+1, -1}, X_test, y_test,
    meta. Deterministic in `seed` (which picks the image subsets, not the
    network init). Cached to `cache_dir`."""
    slug = dataset_slug(hw, n_train, n_test, classes, seed, input_scale)
    cache_path = os.path.join(cache_dir, slug + ".pt") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    if 32 % hw:
        raise ValueError(f"hw must divide 32, got {hw}")
    gen = torch.Generator().manual_seed(seed)
    root = resolve_cifar_root(root)
    x_tr_all, y_tr_all = _load_cifar_split(root, train=True)
    x_te_all, y_te_all = _load_cifar_split(root, train=False)
    x_tr, y_tr, idx_tr = _balanced_subset(x_tr_all, y_tr_all, classes, n_train, gen)
    x_te, y_te, idx_te = _balanced_subset(x_te_all, y_te_all, classes, n_test, gen)
    pool = 32 // hw
    if pool > 1:
        x_tr = F.avg_pool2d(x_tr, pool)
        x_te = F.avg_pool2d(x_te, pool)
    x_tr, x_te = _normalize(x_tr, x_te, input_scale)

    data = {
        "X_train": x_tr.contiguous(), "y_train": y_tr.contiguous(),
        "X_test": x_te.contiguous(), "y_test": y_te.contiguous(),
        "meta": {
            "slug": slug, "hw": hw, "n_train": n_train, "n_test": n_test,
            "seed": seed, "input_scale": input_scale, "classes": tuple(classes),
            "class_names": tuple(CLASS_NAMES[c] for c in classes),
            "train_indices": idx_tr, "test_indices": idx_te,
        },
    }
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cache_path}.tmp.{os.getpid()}"     # several jobs may build it at once
        torch.save(data, tmp)
        os.replace(tmp, cache_path)
    return data


def data_fingerprint(data):
    """Short hash of the tensors, recorded per run to check that runs saw the same data."""
    h = hashlib.sha256()
    for key in ("X_train", "y_train", "X_test", "y_test"):
        h.update(data[key].to(torch.float64).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
