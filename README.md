# Deviations from the NTK Regime in CNNs under Joint Depth-Width Scaling

Code for the paper by Maria Matveev, Pascal Esser, Gitta Kutyniok and Mariia Seleznova.

We study the empirical neural tangent kernel (NTK) of tanh CNNs with $L$ convolutional
layers of $C$ channels, as a function of the depth-to-width ratio $\lambda = L/C$, in the
ordered ($\sigma_w^2 = 1$), critical ($\sigma_w^2 = 3.44$) and chaotic ($\sigma_w^2 = 5$) regimes:

- **NTK dispersion at initialization**: the excess dispersion
  $D_0(x) = \mathbb{E}[\Theta_0(x,x)^2] / \mathbb{E}[\Theta_0(x,x)]^2 - 1$ over independent
  initializations, for fixed CIFAR-10 images.
- **NTK change during training**: the rotation $R_t$ and relative change $\Delta_t$ of
  $\Theta_t(X_P, X_P)$ on a probe set during full-batch gradient descent on binary CIFAR-10
  (airplane vs. frog).

In addition to the CNN, the repository provides code for an MLP of hidden width $M = 9C$ (the same hidden-layer fan-in as a $3 \times 3$ CNN).

Example:

```bash
python src/run_dispersion.py --arch cnn --widths 16 32 --depths 2 8 --n-mc 100 --image 0 --out caches/cnn_k3_img0.pt
python src/run_training.py --arch cnn --widths 16 --depths 4 --sigma-w2 3.44 --seeds 0 --float32 --out-dir results/cnn_k3
```

## Setup

```bash
pip install -r requirements.txt # PyTorch >= 2.0 (torch.func); tested with Python 3.9 and PyTorch 2.5.1
python src/download_cifar.py    # CIFAR-10 into ./data (or set $NTK_CIFAR_ROOT)
python src/verify_ntk.py        # cross-check of the NTK implementations
```

## License

MIT, see `LICENSE`.
