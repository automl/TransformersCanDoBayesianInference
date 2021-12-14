import random

import torch
from torch import nn

from utils import default_device
from .utils import get_batch_to_dataloader


def get_batch(batch_size, seq_len, batch_size_per_gp_sample=None, **config):
    batch_size_per_gp_sample = batch_size_per_gp_sample or batch_size // 16
    assert batch_size % batch_size_per_gp_sample == 0, 'Please choose a batch_size divisible by batch_size_per_gp_sample.'
    num_models = batch_size // batch_size_per_gp_sample
    # standard kaiming uniform init currently...

    models = [config['model']() for _ in range(num_models)]

    sample = sum([[model(seq_len=seq_len) for _ in range(0,batch_size_per_gp_sample)] for model in models],[])

    def normalize_data(data):
        mean = data.mean(0)
        std = data.std(0) + .000001
        eval_xs = (data - mean) / std

        return eval_xs

    x, y = zip(*sample)

    y = torch.stack(y, 1).squeeze(-1).detach()
    x = torch.stack(x, 1).detach()

    x, y = normalize_data(x), y

    return x, y, y


DataLoader = get_batch_to_dataloader(get_batch)
DataLoader.num_outputs = 1

