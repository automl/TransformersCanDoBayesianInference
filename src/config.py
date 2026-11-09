"""Training configuration: typed defaults, YAML loading/saving and command-line overrides.

The dataclasses below are the single source of truth for all default values. A configuration is built by
merging, in this order (later wins):
    1. the defaults of `TrainConfig`
    2. YAML files (only the values that differ from the defaults are needed), e.g. `configs/linear_gp.yaml`
    3. dotted overrides, e.g. `["model.emsize=256", "prior.hyperparameters.noise=0.01"]`
Unknown keys and values of the wrong type raise an error.

Usage in a notebook:
    from src.config import load_config
    cfg = load_config('configs/linear_gp.yaml', overrides=['training.epochs=5'])
    cfg.model.emsize = 256  # dataclasses can also be modified directly
"""
import types
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Literal

import yaml


@dataclass
class GPHyperparameters:
    """Hyperparameters of `src.prior.ExactGPDotProductRBFModel` (see `src.prior.get_model`)."""
    noise: float = .1
    variance: float = .1  # variance of the linear (dot-product) part
    outputscale: float = .1  # RBF output scale; 0 = no RBF part
    lengthscale: float | None = .1  # RBF lengthscale; required if outputscale > 0
    bias: float = 0.  # constant kernel offset (intercept variance); 0 = none


@dataclass
class PriorConfig:
    """Arguments of the prior DataLoader / `src.prior.get_batch`."""
    num_features: int = 1
    equidistant_x: bool = False
    device: str | None = None  # device for sampling the prior; None = `default_device` (use 'cpu' if MPS is unstable)
    hyperparameters: GPHyperparameters = field(default_factory=GPHyperparameters)


@dataclass
class ModelConfig:
    """Arguments of `src.transformer.GaussianTransformerModel`."""
    emsize: int = 512
    nhid: int | None = None  # None = 2 * emsize
    nlayers: int = 6
    nhead: int = 4  # nhead = emsize / 64 in the original paper
    dropout: float = 0.
    input_normalization: bool = False
    min_var: float = 1e-6


@dataclass
class OptimConfig:
    lr: float | None = .001  # None = `get_openai_lr(model)`
    warmup_epochs: int = 50
    aggregate_k_gradients: int = 1


@dataclass
class TrainingConfig:
    epochs: int = 200
    steps_per_epoch: int = 10
    batch_size: int = 1000
    bptt: int = 10  # sequence length (training + query points) of each sampled dataset
    max_eval_pos: int | None = None  # single_eval_pos is sampled from range(max_eval_pos); None = bptt
    eval_pos_sampling: Literal['weighted', 'uniform'] = 'weighted'
    seed: int | None = None  # None = random seed, chosen and recorded by `train`
    device: str | None = None  # device for the model; None = `default_device` (CUDA, MPS, then CPU)


@dataclass
class TrainConfig:
    prior: PriorConfig = field(default_factory=PriorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self):
        """Check constraints between values; raises `ValueError`."""
        m, o, t = self.model, self.optim, self.training
        checks = [
            (m.emsize % m.nhead == 0, f'model.emsize ({m.emsize}) must be divisible by model.nhead ({m.nhead})'),
            (0 <= m.dropout < 1, f'model.dropout must be in [0, 1), got {m.dropout}'),
            (min(t.epochs, t.steps_per_epoch, t.batch_size, o.aggregate_k_gradients) >= 1,
             'training.epochs, training.steps_per_epoch, training.batch_size and optim.aggregate_k_gradients '
             'must be >= 1'),
            (t.steps_per_epoch % o.aggregate_k_gradients == 0,
             'training.steps_per_epoch must be divisible by optim.aggregate_k_gradients'),
            (t.bptt >= 2, f'training.bptt must be >= 2, got {t.bptt}'),
            (t.max_eval_pos is None or 1 <= t.max_eval_pos <= t.bptt,
             f'training.max_eval_pos must be in [1, bptt={t.bptt}], got {t.max_eval_pos}'),
            (self.prior.num_features >= 1, 'prior.num_features must be >= 1'),
            (not self.prior.equidistant_x or self.prior.num_features == 1,
             'prior.equidistant_x requires prior.num_features == 1'),
        ]
        errors = [msg for ok, msg in checks if not ok]
        if errors:
            raise ValueError('Invalid config:\n  ' + '\n  '.join(errors))
        return self


def config_to_dict(config):
    return asdict(config)


def config_from_dict(data):
    """Build and validate a `TrainConfig` from a nested dict; missing keys keep their defaults."""
    return _from_dict(TrainConfig, data, prefix='').validate()


def load_config(*paths, overrides=()):
    """Defaults <- YAML files (in order) <- dotted overrides `key.sub=value` (values parsed as YAML)."""
    data = config_to_dict(TrainConfig())
    for path in paths:
        with open(path) as f:
            _deep_merge(data, yaml.safe_load(f) or {}, prefix='')
    for override in overrides:
        key, sep, value = override.partition('=')
        if not sep:
            raise ValueError(f'Override must have the form key.sub=value, got {override!r}')
        *parents, leaf = key.strip().split('.')
        node = data
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f'Override {override!r}: {p!r} is not a section')
        node[leaf] = yaml.safe_load(value)
    return config_from_dict(data)


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)


def _deep_merge(base, update, prefix):
    if not isinstance(update, dict):
        raise ValueError(f'Expected a mapping at {prefix or "top level"!r}, got {update!r}')
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value, prefix=f'{prefix}{key}.')
        else:
            base[key] = value


def _from_dict(cls, data, prefix):
    if not isinstance(data, dict):
        raise TypeError(f'{prefix.rstrip(".") or "config"}: expected a mapping, got {data!r}')
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f'Unknown config key(s) {[prefix + k for k in unknown]}; '
                         f'valid keys in {prefix.rstrip(".") or "top level"!r}: {sorted(names)}')
    hints = typing.get_type_hints(cls)
    return cls(**{k: _coerce(v, hints[k], prefix + k) for k, v in data.items()})


def _coerce(value, tp, path):
    """Check `value` against the type hint `tp`; ints are accepted for floats, and numeric strings too
    (PyYAML reads e.g. `1e-3` as a string)."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, path)
            except (TypeError, ValueError):
                pass
        raise TypeError(f'{path}: expected {tp}, got {value!r}')
    if origin is Literal:
        if value in typing.get_args(tp):
            return value
        raise ValueError(f'{path}: expected one of {typing.get_args(tp)}, got {value!r}')
    if is_dataclass(tp):
        return value if isinstance(value, tp) else _from_dict(tp, value, prefix=path + '.')
    if tp is float and not isinstance(value, bool):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    elif tp is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    elif tp in (bool, str) and isinstance(value, tp):
        return value
    raise TypeError(f'{path}: expected {tp.__name__}, got {value!r}')
