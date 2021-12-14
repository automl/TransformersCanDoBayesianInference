import random

import torch

from utils import set_locals_in_self
from itertools import repeat
from .prior import PriorDataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats

def get_batch_to_dataloader(get_batch_method_):
    class DL(PriorDataLoader):
        get_batch_method = get_batch_method_

        # Caution, you might need to set self.num_features manually if it is not part of the args.
        def __init__(self, num_steps, fuse_x_y=False, **get_batch_kwargs):
            set_locals_in_self(locals())
            # The stuff outside the or is set as class attribute before instantiation.
            self.num_features = get_batch_kwargs.get('num_features') or self.num_features
            self.num_outputs = get_batch_kwargs.get('num_outputs') or self.num_outputs
            print('DataLoader.__dict__', self.__dict__)

        @staticmethod
        def gbm(*args, fuse_x_y=True, **kwargs):
            x, y, target_y = get_batch_method_(*args, **kwargs)
            if fuse_x_y:
                return torch.cat([x, torch.cat([torch.zeros_like(y[:1]), y[:-1]], 0).unsqueeze(-1).float()],
                                 -1), target_y
            else:
                return (x, y), target_y

        def __len__(self):
            return self.num_steps

        def __iter__(self):
            return iter(self.gbm(**self.get_batch_kwargs, fuse_x_y=self.fuse_x_y) for _ in range(self.num_steps))


    return DL


def plot_features(data, targets):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
    fig2 = plt.figure(constrained_layout=True, figsize=(12, 12))
    spec2 = gridspec.GridSpec(ncols=data.shape[1], nrows=data.shape[1], figure=fig2)
    for d in range(0, data.shape[1]):
        for d2 in range(0, data.shape[1]):
            sub_ax = fig2.add_subplot(spec2[d, d2])
            sub_ax.scatter(data[:, d], data[:, d2],
                           c=targets[:])


def plot_prior(prior):
    s = np.array([prior() for _ in range(0, 10000)])
    count, bins, ignored = plt.hist(s, 50, density=True)
    print(s.min())
    plt.show()

trunc_norm_sampler_f = lambda mu, sigma : lambda: stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
beta_sampler_f = lambda a, b : lambda : np.random.beta(a, b)
gamma_sampler_f = lambda a, b : lambda : np.random.gamma(a, b)
uniform_sampler_f = lambda a, b : lambda : np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b : lambda : np.random.randint(a, b)
zipf_sampler_f = lambda a, b, c : lambda : min(b + np.random.zipf(a), c)
scaled_beta_sampler_f = lambda a, b, scale, minimum : lambda : minimum + round(beta_sampler_f(a, b)() * (scale - minimum + 1) - 0.5)


def normalize_data(data):
    mean = data.mean(0)
    std = data.std(0) + .000001
    data = (data - mean) / std

    return data


def normalize_by_used_features_f(x, num_features_used, num_features):
    return x / (num_features_used / num_features)


class Binarize(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return (x > torch.median(x)).float()


def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)#.reshape(seq_len)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

    return x, y


