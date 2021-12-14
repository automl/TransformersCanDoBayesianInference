import random
import time

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from .utils import get_batch_to_dataloader

def get_batch(batch_size, seq_len, num_features, noisy_std = .1):
    m = torch.normal(0., .1, size=(batch_size,num_features))
    b = 0 # torch.rand(batch_size)
    x = torch.rand(seq_len, batch_size,num_features)
    y_non_noisy = torch.einsum('bf,tbf->tb',m,x)
    y = y_non_noisy + torch.normal(torch.zeros_like(y_non_noisy),noisy_std) # noisy_std is alpha
    return x, y, y_non_noisy

DataLoader = get_batch_to_dataloader(get_batch)
DataLoader.num_outputs = 1


def evaluate(x,y,y_non_noisy, alpha=0.):
    start_time = time.time()
    losses_after_t = [.0]
    for t in range(1,len(x)):
        loss_sum = 0.
        for b_i in range(x.shape[1]):
            clf = Ridge(alpha=alpha)
            clf.fit(x[:t,b_i],y[:t,b_i])
            y_ = clf.predict(x[t,b_i].unsqueeze(0))
            l = nn.MSELoss()(y_non_noisy[t,b_i].unsqueeze(0),torch.tensor(y_))
            loss_sum += l
        losses_after_t.append(loss_sum/x.shape[1])
    return torch.tensor(losses_after_t), time.time()-start_time

if __name__ == '__main__':
    for alpha in [.001,.01,.5,1.]:
        print(alpha, evaluate(*get_batch(1000,10,noisy_std=.01),alpha=alpha))
