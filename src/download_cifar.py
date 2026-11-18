"""One-time CIFAR-10 download 

    python src/download_cifar.py [--root DIR]      # default: ./data
"""

import argparse

from torchvision import datasets

import data as data_mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=data_mod.DEFAULT_CIFAR_ROOT)
    args = ap.parse_args()
    datasets.CIFAR10(root=args.root, train=True, download=True)
    datasets.CIFAR10(root=args.root, train=False, download=True)
    print(f"CIFAR-10 is in {args.root}")


if __name__ == "__main__":
    main()
