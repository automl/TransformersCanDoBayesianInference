"""GP prior (fixed hyperparameters) for training a PFN.

`get_batch` / `DataLoader` sample from `ExactGPDotProductRBFModel` (linear trend + optional RBF part).
"""

import torch
import gpytorch
from src.utils import get_batch_to_dataloader, default_device


class ExactGPDotProductRBFModel(gpytorch.models.ExactGP):
    """Exact GP with constant mean, dot-product + scaled RBF kernel and Gaussian likelihood.

    Kernel: `k(x, x') = bias + variance * x^T x' + outputscale * exp(-||x - x'||^2 / (2 * lengthscale^2))`,
    i.e. a random linear trend `b + w^T x` with `b ~ N(0, bias)`, `w ~ N(0, variance * I)` (sklearn's
    `DotProduct` with `sigma_0**2 = bias`, scaled by `variance`) plus independent smooth RBF deviations.
    The linear part needs no `ScaleKernel` (its `variance` already is the amplitude); the RBF part does.

    All hyperparameters are fixed at construction time through the `initial_value` of each
    parameter's constraint (gpytorch applies it when the constraint is registered), so no
    attributes need to be edited after initialization.

    :param train_x: training inputs, shape (batch, n, num_features), or None for a prior-only model
    :param train_y: training targets, shape (batch, n), or None for a prior-only model.
        A prior-only model must be put in eval mode (`model.eval()`) before calling it.
    :param noise: observation noise variance of the Gaussian likelihood
    :param variance: variance of the linear weights (scale of `x^T x'`), must be > 0
    :param outputscale: output scale (signal variance) of the RBF part; `0` means no RBF part (pure linear model)
    :param lengthscale: lengthscale of the RBF kernel (shared across features); required if `outputscale > 0`
    :param bias: constant offset of the kernel (intercept variance); `0` means no offset
    :param min_noise: lower bound of the noise constraint
    """

    def __init__(self, train_x, train_y, noise, variance, outputscale=0., lengthscale=None, bias=0.,
                 min_noise=1.e-9):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(min_noise, initial_value=noise)
        )

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.LinearKernel(
            variance_constraint=gpytorch.constraints.Positive(initial_value=variance)
        )

        # Positive constraints cannot represent 0, so the RBF and offset kernels are only added if needed.
        if outputscale > 0:
            if lengthscale is None:
                raise ValueError('`lengthscale` is required when `outputscale > 0`.')
            self.covar_module = self.covar_module + gpytorch.kernels.ScaleKernel(
                base_kernel=gpytorch.kernels.RBFKernel(
                    lengthscale_constraint=gpytorch.constraints.Positive(initial_value=lengthscale)
                ),
                outputscale_constraint=gpytorch.constraints.Positive(initial_value=outputscale)
            )

        if bias > 0:
            self.covar_module = self.covar_module + gpytorch.kernels.ConstantKernel(
                constant_constraint=gpytorch.constraints.Positive(initial_value=bias)
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_model(x, y, hyperparameters):
    """Get the `ExactGPDotProductRBFModel` and its likelihood.

    :param hyperparameters: dict with `noise`, `variance` and optionally `outputscale` (default 0, no RBF part),
        `lengthscale` (required if `outputscale > 0`) and `bias` (default 0)
    """

    model = ExactGPDotProductRBFModel(
        train_x=x,
        train_y=y,
        noise=hyperparameters["noise"],
        variance=hyperparameters["variance"],
        outputscale=hyperparameters.get("outputscale", 0.),
        lengthscale=hyperparameters.get("lengthscale"),
        bias=hyperparameters.get("bias", 0.))

    return model, model.likelihood


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
    """Sample `batch_size` datasets from the `ExactGPDotProductRBFModel` prior (see `get_model` for the
    hyperparameters).

    :return: x: (seq_len, batch_size, num_features), y and target_y (identical): (seq_len, batch_size)
    """
    if hyperparameters is None:
        hyperparameters = {
            "noise": .1,
            "variance": .1,
            "outputscale": .1,
            "lengthscale": .1,
            "bias": 0.}

    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations', (True, True, True))):

        # -> (batch_size, seq_len, num_feat)
        if equidistant_x:
            assert num_features == 1
            x = torch.linspace(0, 1, seq_len, device=device)[:, None].repeat(batch_size, 1, 1)
        else:
            x = torch.rand(batch_size, seq_len, num_features, device=device)

        # No training data: in eval mode the model returns the GP prior N(m(x), K(x, x)) for any x.
        # (In train mode a model without training data would raise an error.)
        model, likelihood = get_model(None, None, hyperparameters)
        model.to(device)
        model.eval()

        # prior_mode is redundant without training data, but makes the intent explicit.
        with gpytorch.settings.prior_mode(True):
            d = model(x)  # prior of the noise-free function f
            d = likelihood(d)  # adds observation noise -> prior of y
            sample = d.sample().transpose(0, 1)

    # -> (seq_len, batch_size, num_feat), (seq_len, batch_size), (seq_len, batch_size)
    return x.transpose(0, 1), sample, sample  # x.shape = (T,B,H)


DataLoader = get_batch_to_dataloader(get_batch)
