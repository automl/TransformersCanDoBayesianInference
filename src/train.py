"""Minimal PFN training on the GP prior (`src/prior.py`).

Refactored copy of `train.py` with fixed choices:
    encoder:      nn.Linear(num_features, emsize)
    y_encoder:    nn.Linear(1, emsize)
    pos_encoder:  NoPositionalEncoding
    criterion:    `model.criterion` = nn.GaussianNLLLoss(reduction='none', full=True); the model returns
                  (mean, variance) for every query point (variance mapped to be positive inside the model)

Run from the repository root, e.g.:
    python -m src.train --num_features 1 --epochs 10 --bptt 20 --emsize 128 --nhead 2
"""
import argparse
import time

import torch
import yaml
from torch import nn

from src import prior
from src.positional_encodings import NoPositionalEncoding
from src.transformer import TransformerModel
from src.utils import (
    default_device,
    get_cosine_schedule_with_warmup,
    get_openai_lr,
    get_uniform_single_eval_pos_sampler,
    get_weighted_single_eval_pos_sampler,
)


def train(
        priordataloader_class=prior.DataLoader,
        emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.2,
        epochs=10, steps_per_epoch=100, batch_size=200,
        bptt=10,
        lr=None, warmup_epochs=10,
        input_normalization=False, decoder=None, extra_prior_kwargs_dict=None,
        scheduler=get_cosine_schedule_with_warmup, load_weights_from_this_state_dict=None,
        single_eval_pos_gen=None, device=None, aggregate_k_gradients=1, verbose=True):
    """Train a PFN with Gaussian NLL loss.

    :param priordataloader_class: DataLoader class of the prior (default: GP prior from `src/prior.py`)
    :param bptt: sequence length (training + query points) of each sampled dataset
    :param extra_prior_kwargs_dict: extra kwargs for the prior DataLoader, forwarded to `get_batch`,
        e.g. `{'num_features': 1, 'hyperparameters': {...}}`. `num_features` is required.
    :param single_eval_pos_gen: int or callable returning the number of training points per batch,
        must be in `[0, bptt)`. Default: `get_weighted_single_eval_pos_sampler(bptt)`.
    :param device: device for the model; default `default_device` (CUDA, MPS, then CPU)
    :return: (mean loss of the last epoch, mean loss per single_eval_pos of the last epoch, model on CPU)
    """

    device = device or default_device
    print(f'Using {device} device')

    extra_prior_kwargs_dict = extra_prior_kwargs_dict or {}
    if single_eval_pos_gen is None:
        single_eval_pos_gen = get_weighted_single_eval_pos_sampler(bptt)

    dl = priordataloader_class(
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        seq_len=bptt,
        **extra_prior_kwargs_dict)

    model = TransformerModel(
        encoder=nn.Linear(dl.num_features, emsize),
        ninp=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout,
        y_encoder=nn.Linear(1, emsize),
        pos_encoder=NoPositionalEncoding(emsize, bptt * 2),  # arguments don't matter
        decoder=decoder,
        input_normalization=input_normalization)

    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    model.to(device)

    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = scheduler(optimizer, warmup_epochs, epochs)

    def train_epoch():
        model.train()
        total_loss = 0.
        total_positional_losses = torch.zeros(bptt)
        total_positional_losses_recorded = torch.zeros(bptt)
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, \
            'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'

        for batch, ((x, y), targets) in enumerate(dl):
            time_to_get_batch = time.time() - before_get_batch
            before_forward = time.time()

            single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
            assert 0 <= single_eval_pos < bptt, f'single_eval_pos must be in [0, {bptt}), got {single_eval_pos}.'

            # each (bptt - single_eval_pos, batch_size)
            mean_pred, var_pred = model((x.to(device), y.to(device)), single_eval_pos=single_eval_pos)
            forward_time = time.time() - before_forward

            targets = targets[single_eval_pos:].to(device)
            losses = model.criterion(mean_pred, targets, var=var_pred)  # (num_queries, batch_size)

            loss = losses.mean()
            loss.backward()
            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

            step_time = time.time() - before_forward

            total_loss += loss.item()
            total_positional_losses[single_eval_pos] += loss.item()
            total_positional_losses_recorded[single_eval_pos] += 1

            before_get_batch = time.time()

        # positions that were never sampled in this epoch are nan
        positional_losses = (total_positional_losses / total_positional_losses_recorded).tolist()
        return total_loss / steps_per_epoch, positional_losses, time_to_get_batch, forward_time, step_time

    total_loss, total_positional_losses = float('inf'), None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time = train_epoch()

        if verbose:
            print('-' * 89)
            print(
                f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                f' forward time {forward_time:5.2f}')
            print('-' * 89)

        scheduler.step()

    return total_loss, total_positional_losses, model.to('cpu')


def _parse_args(config_parser, parser):
    # Values from a YAML config file (--config) become defaults; command line arguments override them.
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
    config_parser.add_argument('--config')
    parser = argparse.ArgumentParser()

    # GP prior (forwarded to `src.prior.get_batch` via `extra_prior_kwargs_dict`)
    parser.add_argument('--num_features', default=1, type=int)
    parser.add_argument('--noise', default=.1, type=float)
    parser.add_argument('--variance', default=.1, type=float, help='Variance of the linear (dot-product) part.')
    parser.add_argument('--bias', default=0., type=float, help='Constant kernel offset; 0 = none.')
    parser.add_argument('--outputscale', default=.1, type=float, help='RBF output scale; 0 = no RBF part.')
    parser.add_argument('--lengthscale', default=.1, type=float)
    parser.add_argument('--prior_device', default=None, type=str,
                        help='Device for sampling the GP prior; use "cpu" if sampling is unstable on MPS.')

    parser.add_argument('--bptt', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--max_eval_pos', default=None, type=int,
                        help='single_eval_pos is sampled from range(max_eval_pos); default: bptt.')
    parser.add_argument('--eval_pos_sampling', default='weighted', choices=['weighted', 'uniform'])

    # these can likely be mostly left at defaults
    parser.add_argument('--emsize', default=512, type=int)  # sometimes even larger is better e.g. 1024
    parser.add_argument('--nlayers', default=6, type=int)
    parser.add_argument('--nhid', default=None, type=int)  # 2*emsize is the default
    parser.add_argument('--nhead', default=4, type=int)  # nhead = emsize / 64 in the original paper
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', '--learning_rate', default=.001, type=float)  # try also .0003, .0001

    args, _ = _parse_args(config_parser, parser)
    kwargs = dict(args.__dict__)

    if kwargs['nhid'] is None:
        kwargs['nhid'] = 2 * kwargs['emsize']

    extra_prior_kwargs_dict = {
        'num_features': kwargs.pop('num_features'),
        'hyperparameters': {k: kwargs.pop(k) for k in ('noise', 'variance', 'bias', 'outputscale', 'lengthscale')},
    }
    prior_device = kwargs.pop('prior_device')
    if prior_device is not None:
        extra_prior_kwargs_dict['device'] = prior_device
    kwargs['extra_prior_kwargs_dict'] = extra_prior_kwargs_dict

    max_eval_pos = kwargs.pop('max_eval_pos') or kwargs['bptt']
    get_sampler = {'weighted': get_weighted_single_eval_pos_sampler,
                   'uniform': get_uniform_single_eval_pos_sampler}[kwargs.pop('eval_pos_sampling')]
    kwargs['single_eval_pos_gen'] = get_sampler(max_eval_pos)

    print("ARGS for `train`:", kwargs)
    train(prior.DataLoader, **kwargs)
