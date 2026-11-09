"""GP prior for training a PFN.

`get_batch` / `DataLoader` sample from the `ExactGPDotProductRBFModel` prior (linear trend + optional RBF
part); every dataset can get its own hyperparameters drawn from a distribution (`sample_hyperparameters`).
The gpytorch models (fixed hyperparameters) are used for exact GP posteriors and the Beta-Binomial prior.
`ExactGPDotProductRBFMultitaskModel` is the multi-output (ICM) extension of `ExactGPDotProductRBFModel`; `get_batch_beta_binomial` uses it
(2 outputs) to generate counts `(k, n)` for the Beta-Binomial PFN head.
"""

import math

import torch
import gpytorch
import gpytorch.kernels as gk
import gpytorch.constraints as gc
from linear_operator.utils.cholesky import psd_safe_cholesky
from src.utils import get_batch_to_dataloader, default_device


def dot_product_rbf_kernel(variance, outputscale=0., lengthscale=None, bias=0.):
    """Kernel `bias + variance * x^T x' + outputscale * RBF(lengthscale)` with fixed hyperparameters
    (see `ExactGPDotProductRBFModel`)."""
    covar_module = gk.LinearKernel(
        variance_constraint=gc.Positive(initial_value=variance)
    )

    # Positive constraints cannot represent 0, so the RBF and offset kernels are only added if needed.
    if outputscale > 0:
        if lengthscale is None:
            raise ValueError('`lengthscale` is required when `outputscale > 0`.')
        covar_module = covar_module + gk.ScaleKernel(
            base_kernel=gk.RBFKernel(lengthscale_constraint=gc.Positive(initial_value=lengthscale)),
            outputscale_constraint=gc.Positive(initial_value=outputscale)
        )

    if bias > 0:
        covar_module = covar_module + gk.ConstantKernel(
            constant_constraint=gc.Positive(initial_value=bias)
        )

    return covar_module


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
            noise_constraint=gc.GreaterThan(min_noise, initial_value=noise)
        )

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = dot_product_rbf_kernel(variance, outputscale, lengthscale, bias)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPDotProductRBFMultitaskModel(gpytorch.models.ExactGP):
    """Multi-output (Intrinsic Coregionalization Model, ICM) version of `ExactGPDotProductRBFModel`.

    The output at each input is a vector `y(x)` of `num_tasks` components with covariance

        Cov(y_i(x), y_j(x')) = k(x, x') * B_ij + noise * delta_ij * delta(x, x')

    where `k` is the dot-product + RBF kernel of `ExactGPDotProductRBFModel` (shared by all outputs) and
    `B` is the `num_tasks x num_tasks` task covariance matrix (PSD). `B = I` gives independent outputs,
    off-diagonal entries correlate them. Each output has a constant mean.

    gpytorch parametrizes `B = W W^T + diag(v)` (`IndexKernel`); here `W` is set from the eigendecomposition
    of `task_covariance` (rank `num_tasks`, also works for singular `B`) and `v = min_task_var`, so
    `B ~= task_covariance`. All hyperparameters are fixed at construction time.

    :param train_x: training inputs, shape (batch, n, num_features), or None for a prior-only model
    :param train_y: training targets, shape (batch, n, num_tasks), or None for a prior-only model.
        A prior-only model must be put in eval mode (`model.eval()`) before calling it.
    :param num_tasks: number of outputs
    :param task_covariance: PSD matrix `B`, shape (num_tasks, num_tasks); default identity
    :param noise: observation noise variance (shared by all outputs, independent across outputs)
    :param variance, outputscale, lengthscale, bias: see `ExactGPDotProductRBFModel`
    :param min_noise: lower bound of the noise constraint
    :param min_task_var: diagonal `v` added to `W W^T` (gpytorch requires `v > 0`)
    """

    def __init__(self, train_x, train_y, num_tasks, noise, variance, outputscale=0., lengthscale=None, bias=0.,
                 task_covariance=None, min_noise=1.e-9, min_task_var=1.e-9):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks,
            has_task_noise=False,
            noise_constraint=gc.GreaterThan(min_noise, initial_value=noise)
        )

        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        self.covar_module = gk.MultitaskKernel(
            dot_product_rbf_kernel(variance, outputscale, lengthscale, bias), num_tasks=num_tasks
        )
        # MultitaskKernel does not forward `var_constraint`, so its IndexKernel is replaced.
        self.covar_module.task_covar_module = gk.IndexKernel(
            num_tasks=num_tasks,
            rank=num_tasks,
            var_constraint=gc.Positive(initial_value=min_task_var)
        )
        self.covar_module.task_covar_module.initialize(
            covar_factor=self.task_covar_factor(task_covariance, num_tasks))

    @staticmethod
    def task_covar_factor(task_covariance, num_tasks):
        """`W` with `W W^T = task_covariance` (symmetric PSD), shape (num_tasks, num_tasks)."""
        if task_covariance is None:
            return torch.eye(num_tasks)

        B = torch.as_tensor(task_covariance, dtype=torch.get_default_dtype())
        if B.shape != (num_tasks, num_tasks) or not torch.allclose(B, B.T):
            raise ValueError(f'`task_covariance` must be a symmetric ({num_tasks}, {num_tasks}) matrix.')

        eigvals, eigvecs = torch.linalg.eigh(B)
        if eigvals.min() < -1e-6 * eigvals.abs().max():
            raise ValueError('`task_covariance` must be positive semi-definite.')

        return eigvecs * eigvals.clamp_min(0).sqrt()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


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
    """Prior covariance of `ExactGPDotProductRBFModel` with a different set of hyperparameters per dataset.

    `K_b = bias_b + variance_b * x_b x_b^T + outputscale_b * exp(-||x_b - x_b'||^2 / (2 * lengthscale_b^2))`
    (the same kernel as `dot_product_rbf_kernel`, without noise).

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
    """Sample `batch_size` datasets from the `ExactGPDotProductRBFModel` prior (zero mean).

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


def get_model_beta_binomial(x, y, hyperparameters):
    """Get the 2-output `ExactGPDotProductRBFMultitaskModel` (latent GP of `get_batch_beta_binomial`)
    and its likelihood.

    :param hyperparameters: dict with `noise`, `variance` and optionally `outputscale`, `lengthscale`, `bias`
        (see `get_model`) and `task_covariance` (2x2 PSD matrix, default identity)
    """

    model = ExactGPDotProductRBFMultitaskModel(
        train_x=x,
        train_y=y,
        num_tasks=2,
        noise=hyperparameters["noise"],
        variance=hyperparameters["variance"],
        outputscale=hyperparameters.get("outputscale", 0.),
        lengthscale=hyperparameters.get("lengthscale"),
        bias=hyperparameters.get("bias", 0.),
        task_covariance=hyperparameters.get("task_covariance"))

    return model, model.likelihood


@torch.no_grad()
def get_batch_beta_binomial(
        batch_size,
        seq_len: int,
        num_features: int,
        device=default_device,
        hyperparameters: dict | None = None,
        equidistant_x: bool = False
):
    """Sample `batch_size` datasets of counts `(k, n)` with `0 <= k <= n`, driven by a 2-output GP.

    Generative process for every input `x` (all `seq_len` points of a dataset share one GP sample):

        (f_1(x), f_2(x)) ~ ExactGPDotProductRBFMultitaskModel (ICM, `num_tasks=2`, incl. noise)
        p(x)      = sigmoid(f_1(x))                                   success probability
        rate(x)   = min(total_count_rate * exp(f_2(x)), max_rate)     Poisson rate (log link)
        n(x)      = min_total_count + Poisson(rate(x))                number of trials
        k(x)      ~ Binomial(n(x), p(x))                              number of successes

    `task_covariance` correlates `f_1` and `f_2`, i.e. the success probability and the number of trials.

    :param hyperparameters: GP hyperparameters (see `get_model_beta_binomial`) and optionally
        `total_count_rate` (Poisson rate at `f_2 = 0`, default 10), `min_total_count` (default 1; use 0 to
        allow `n = 0`) and `max_rate` (default 1e4, guards against overflow of `exp`)
    :return: x: (seq_len, batch_size, num_features), y and target_y (identical): (seq_len, batch_size, 2) with
        `[..., 0] = k` and `[..., 1] = n` (float tensors with integer values)
    """
    if hyperparameters is None:
        hyperparameters = {
            "noise": .01,
            "variance": 1.,
            "outputscale": 1.,
            "lengthscale": .2,
            "bias": .5,
            "task_covariance": [[1., 0.], [0., 1.]],
            "total_count_rate": 10.,
            "min_total_count": 1,
            "max_rate": 1e4}

    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations', (True, True, True))):

        # -> (batch_size, seq_len, num_feat)
        if equidistant_x:
            assert num_features == 1
            x = torch.linspace(0, 1, seq_len, device=device)[:, None].repeat(batch_size, 1, 1)
        else:
            x = torch.rand(batch_size, seq_len, num_features, device=device)

        model, likelihood = get_model_beta_binomial(None, None, hyperparameters)
        model.to(device)
        model.eval()

        with gpytorch.settings.prior_mode(True):
            f = likelihood(model(x)).sample()  # (batch_size, seq_len, 2)

    p = torch.sigmoid(f[..., 0])
    rate = (
            hyperparameters.get('total_count_rate', 10.) *
            torch.exp(f[..., 1])
    ).clamp_max(hyperparameters.get('max_rate', 1e4))

    # Poisson/Binomial sampling on CPU (not implemented on all backends, e.g. MPS).
    n = hyperparameters.get('min_total_count', 1) + torch.poisson(rate.cpu())
    k = torch.distributions.Binomial(total_count=n, probs=p.cpu()).sample()

    sample = torch.stack([k, n], dim=-1).to(device)

    # -> (seq_len, batch_size, num_feat), (seq_len, batch_size, 2), (seq_len, batch_size, 2)
    return (
        x.transpose(0, 1),
        sample.transpose(0, 1),
        sample.transpose(0, 1)
    )


DataLoader = get_batch_to_dataloader(get_batch)
