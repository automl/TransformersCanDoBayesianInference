"""GP prior for training a PFN.

`get_batch` / `DataLoader` sample datasets from a zero-mean GP with kernel
`bias + variance * x^T x' + outputscale * RBF(lengthscale)` (random linear trend + optional smooth deviations)
and Gaussian observation noise. Every dataset can get its own hyperparameters drawn from a distribution
(`sample_hyperparameters`).
"""

import math

import torch
from linear_operator.utils.cholesky import psd_safe_cholesky
from src.utils import get_batch_to_dataloader, default_device


def sample_hyperparameters(hyperparameters, batch_size, device=None):
    """Draw one value of every hyperparameter for each of the `batch_size` datasets.

    Every entry of `hyperparameters` is either
      - a number (the same value for all datasets),
      - `None` (kept as `None`), or
      - a distribution `{'dist': 'uniform' | 'log_uniform', 'low': a, 'high': b}`, sampled independently per
        dataset. `log_uniform` (uniform in `log`, requires `a > 0`) suits scale parameters spanning orders of
        magnitude.

    :return: dict with the same keys; numbers and distributions become tensors of shape (batch_size,)
    """
    sampled = {}
    for name, spec in hyperparameters.items():
        if spec is None:
            sampled[name] = None
        elif isinstance(spec, dict):
            low, high, dist = float(spec['low']), float(spec['high']), spec['dist']
            u = torch.rand(batch_size, device=device)
            if dist == 'uniform':
                sampled[name] = low + (high - low) * u
            elif dist == 'log_uniform':
                sampled[name] = torch.exp(math.log(low) + (math.log(high) - math.log(low)) * u)
            else:
                raise ValueError(f'{name}: unknown distribution {dist!r} (use uniform or log_uniform)')
        else:
            sampled[name] = torch.full((batch_size,), float(spec), device=device)
    return sampled


def dot_product_rbf_covariance(x, variance, outputscale=None, lengthscale=None, bias=None):
    """Prior covariance (without noise) with a different set of hyperparameters per dataset.

    `K_b = bias_b + variance_b * x_b x_b^T + outputscale_b * exp(-||x_b - x_b'||^2 / (2 * lengthscale_b^2))`,
    i.e. a random linear trend `c + w^T x` with `c ~ N(0, bias)`, `w ~ N(0, variance * I)` plus independent
    smooth RBF deviations.

    :param x: inputs, shape (batch_size, n, num_features)
    :param variance, outputscale, lengthscale, bias: tensors of shape (batch_size,); `outputscale` and `bias`
        may be `None` (= 0)
    :return: covariance matrices, shape (batch_size, n, n)
    """
    def b(v):
        return v[:, None, None]

    covar = b(variance) * (x @ x.transpose(-1, -2))
    if bias is not None:
        covar = covar + b(bias)
    if outputscale is not None and bool((outputscale > 0).any()):
        if lengthscale is None:
            raise ValueError('`lengthscale` is required when `outputscale > 0`.')
        sq_dist = (x[:, :, None, :] - x[:, None, :, :]).square().sum(-1)
        covar = covar + b(outputscale) * torch.exp(-.5 * sq_dist / b(lengthscale).square())
    return covar


@torch.no_grad()
# WARNING: `default_device` is 'mps' on Apple Silicon. MPS has no float64 support, so GP prior sampling
# can be numerically unstable for very small noise (e.g. noise=1e-4). If you see NotPSD/jitter
# warnings or errors, pass device='cpu' here.
def get_batch(
        batch_size,
        seq_len: int,
        num_features: int,
        device=default_device,
        hyperparameters: dict | None = None,
        equidistant_x: bool = False
):
    """Sample `batch_size` datasets from the GP prior (zero mean, kernel of `dot_product_rbf_covariance`).

    Every dataset gets its own hyperparameters `noise`, `variance`, `outputscale`, `lengthscale`, `bias`
    drawn by `sample_hyperparameters`, so a hyperparameter given as a distribution varies from dataset to
    dataset (fixed numbers reproduce a single GP prior). Missing `outputscale`/`bias` default to 0.

    :return: x: (seq_len, batch_size, num_features), y and target_y (identical): (seq_len, batch_size)
    """
    if hyperparameters is None:
        hyperparameters = {
            "noise": .1,
            "variance": .1,
            "outputscale": .1,
            "lengthscale": .1,
            "bias": 0.}

    # -> (batch_size, seq_len, num_feat)
    if equidistant_x:
        assert num_features == 1
        x = torch.linspace(0, 1, seq_len, device=device)[:, None].repeat(batch_size, 1, 1)
    else:
        x = torch.rand(batch_size, seq_len, num_features, device=device)

    hp = sample_hyperparameters(hyperparameters, batch_size, device=device)
    covar = dot_product_rbf_covariance(
        x, hp['variance'], hp.get('outputscale'), hp.get('lengthscale'), hp.get('bias'))
    # Observation noise on the diagonal -> covariance of y.
    covar = covar + hp['noise'][:, None, None] * torch.eye(seq_len, device=device)

    chol = psd_safe_cholesky(covar)  # adds jitter if needed
    sample = (chol @ torch.randn(batch_size, seq_len, 1, device=device)).squeeze(-1).transpose(0, 1)

    # -> (seq_len, batch_size, num_feat), (seq_len, batch_size), (seq_len, batch_size)
    return x.transpose(0, 1), sample, sample  # x.shape = (T,B,H)


DataLoader = get_batch_to_dataloader(get_batch)
