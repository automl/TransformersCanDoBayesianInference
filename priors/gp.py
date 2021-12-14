import time
import random

import numpy as np
import torch
from torch import nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from .utils import get_batch_to_dataloader


length_scale_sampling_gp = .6

def get_gp(length_scale=None):
    return GaussianProcessRegressor(
        kernel=RBF(length_scale=length_scale or length_scale_sampling_gp, length_scale_bounds='fixed'),
        random_state=0, optimizer=None)


def get_batch(batch_size, seq_len, num_features, noisy_std=None):
    # m = torch.normal(0.,.1,size=(batch_size,num_features))
    # m2 = torch.rand(batch_size,num_features)
    # b = 0 # torch.rand(batch_size)
    x_t = torch.rand(batch_size, seq_len, num_features)
    # gp_b = TensorGP(kernel=TensorRBF(noisy_std))
    # y_t = gp_b.sample_from_GP_prior(x_t).detach()

    gpr = get_gp(noisy_std)
    y_t = torch.zeros(batch_size, seq_len)

    for i in range(len(y_t)):
        y_t[i] += gpr.sample_y(x_t[i], random_state=random.randint(0, 2 ** 32)).squeeze()
    x, y = x_t.transpose(0, 1), y_t.transpose(0, 1)
    # x, _ = torch.sort(x,dim=0)
    return x, y, y


DataLoader = get_batch_to_dataloader(get_batch)
DataLoader.num_outputs = 1

def evaluate(x, y, y_non_noisy, use_mse=False, length_scale=length_scale_sampling_gp):
    start_time = time.time()
    losses_after_t = [.0]
    for t in range(1, len(x)):
        loss_sum = 0.
        for b_i in range(x.shape[1]):
            gpr = get_gp(length_scale).fit(x[:t, b_i], y[:t, b_i])
            means, stds = gpr.predict(x[t, b_i].unsqueeze(0), return_std=True)
            assert len(means) == 1 == len(stds)
            if use_mse:
                c = nn.MSELoss()
                l = c(torch.tensor(means), y[t, b_i].unsqueeze(-1))
            else:
                c = nn.GaussianNLLLoss(full=True)
                l = c(torch.tensor(means), y[t, b_i].unsqueeze(-1),
                      var=torch.tensor(stds) ** 2)
            loss_sum += l


        losses_after_t.append(loss_sum / x.shape[1])

    return torch.tensor(losses_after_t), time.time()-start_time

if __name__ == '__main__':
    ls = .1
    for alpha in set([ls, ls * 1.1, ls * .9]):
        print(alpha)
        for redo_idx in range(1):
            print(
                evaluate(*get_batch(1000, 10, noisy_std=ls, num_features=10), use_mse=False, length_scale=alpha))
