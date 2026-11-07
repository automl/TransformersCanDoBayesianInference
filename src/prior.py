"""GP prior (fixed hyperparameters) for training a PFN.
"""

import torch
import gpytorch
from src.utils import get_batch_to_dataloader, default_device


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP with constant mean, scaled RBF kernel and Gaussian likelihood.

    All hyperparameters are fixed at construction time through the `initial_value` of each
    parameter's constraint (gpytorch applies it when the constraint is registered), so no
    attributes need to be edited after initialization.

    :param train_x: training inputs, shape (batch, n, num_features), or None for a prior-only model
    :param train_y: training targets, shape (batch, n), or None for a prior-only model.
        A prior-only model must be put in eval mode (`model.eval()`) before calling it.
    :param noise: observation noise variance of the Gaussian likelihood
    :param outputscale: output scale (signal variance) of the ScaleKernel
    :param lengthscale: lengthscale of the RBF kernel (shared across features)
    :param min_noise: lower bound of the noise constraint
    """

    def __init__(self, train_x, train_y, noise, outputscale, lengthscale, min_noise=1.e-9):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(min_noise, initial_value=noise)
        )

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Positive(initial_value=lengthscale)
            ),
            outputscale_constraint=gpytorch.constraints.Positive(initial_value=outputscale)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_model(x, y, hyperparameters):
    """ Get model and likelihood """

    model = ExactGPModel(
        train_x=x,
        train_y=y,
        noise=hyperparameters["noise"],
        outputscale=hyperparameters["outputscale"],
        lengthscale=hyperparameters["lengthscale"])

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
    if hyperparameters is None:
        hyperparameters = {
            "noise": .1,
            "outputscale": .1,
            "lengthscale": .1}

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
