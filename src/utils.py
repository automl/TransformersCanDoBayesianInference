"""Utilities needed by `src/prior.py`, `src/transformer.py` and `src/train.py`.

Copied from:
  - `utils.py`: `get_default_device`, `default_device`, `SeqBN`, `get_cosine_schedule_with_warmup`,
    `get_openai_lr`, `get_weighted_single_eval_pos_sampler`, `get_uniform_single_eval_pos_sampler`
  - `priors/prior.py`: `PriorDataLoader`
  - `priors/utils.py`: `get_batch_to_dataloader`
"""
import math
import random

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """Learning rate schedule: linear warmup from 0 to the optimizer's lr over `num_warmup_steps`,
    then cosine decay to 0 (for the default `num_cycles=0.5`) at `num_training_steps`.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_openai_lr(transformer_model):
    """Heuristic maximum learning rate as a function of the number of parameters."""
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)


def get_weighted_single_eval_pos_sampler(max_len):
    """
    Sampler for `single_eval_pos` in `range(max_len)`; position i has weight 1 / (max_len - i), so
    long contexts are sampled more often. At most `max_len` - 1 examples are shown to the Transformer.
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(max_len), [1 / (max_len - i) for i in range(max_len)])[0]


def get_uniform_single_eval_pos_sampler(max_len):
    """
    Sampler for `single_eval_pos`, uniform in `range(max_len)`.
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(max_len))[0]


class SeqBN(nn.Module):
    """BatchNorm1d over the embedding dim, applied to all (seq_len * batch_size) tokens jointly."""

    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


def get_default_device():
    """Return the best available device: CUDA (NVIDIA), then MPS (Apple Silicon GPU), then CPU.

    WARNING (MPS / Apple Silicon):
      - MPS does not support float64. GP code (gpytorch Cholesky) may become numerically unstable
        for very small noise levels (e.g. noise=1e-4) and raise NotPSD/jitter warnings or errors.
        If that happens, sample the GP prior on CPU by passing device='cpu' to `get_batch`.
      - Some ops are not implemented on MPS; set PYTORCH_ENABLE_MPS_FALLBACK=1 to fall back to CPU.
    """
    if torch.cuda.is_available():
        return 'cuda:0'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu:0'


default_device = get_default_device()


class PriorDataLoader(DataLoader):
    """Interface for prior data loaders consumed by `train.train`.

    Subclasses do not call `DataLoader.__init__`; they only implement `__len__` and `__iter__`
    (yielding `(data, targets)` batches sampled on the fly). Required interface:
      - `__init__(num_steps, num_features, **get_batch_kwargs)`
      - instance attribute: `num_features: int`
      - batches are `((x, y), target_y)` with x: (seq_len, batch_size, num_features),
        y and target_y: (seq_len, batch_size), i.e. a single scalar output per point
      - optional: `validate(model)` called periodically during training
    See `get_batch_to_dataloader` for the standard implementation.
    """


def get_batch_to_dataloader(get_batch_method_):
    """Wrap a prior's `get_batch` function into a DataLoader-like class.

    Every iteration over an instance calls `get_batch_method_(num_features=..., **get_batch_kwargs)`
    `num_steps` times, producing a fresh synthetic batch each time (nothing is cached).

    Each step yields `((x, y), target_y)`; only priors with a single scalar output per point are supported.
    `num_features` is a required constructor argument: it is stored on the instance (read by `train.py`)
    and forwarded to `get_batch`.

    :param get_batch_method_: function `(batch_size, seq_len, num_features, ...) -> (x, y, target_y)` with
        x: (seq_len, batch_size, num_features), y and target_y: (seq_len, batch_size)
    """

    class DL(PriorDataLoader):
        def __init__(self, num_steps, num_features, **get_batch_kwargs):
            # torch's DataLoader.__init__ is intentionally not called: there is no dataset/sampler,
            # batches are generated on the fly in `__iter__`. Only `__len__`/`__iter__` are supported.
            self.num_steps = num_steps
            self.num_features = num_features
            self.get_batch_kwargs = {'num_features': num_features, **get_batch_kwargs}

        @staticmethod
        def gbm(*args, **kwargs):
            x, y, target_y = get_batch_method_(*args, **kwargs)
            return (x, y), target_y

        def __len__(self):
            return self.num_steps

        def __iter__(self):
            return iter(self.gbm(**self.get_batch_kwargs) for _ in range(self.num_steps))

    return DL
