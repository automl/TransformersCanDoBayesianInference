import torch
from torch import nn


class ScaledSoftmaxCE(nn.Module):
    def forward(self, x, label):
        logits = x[..., :-10]
        temp_scales = x[..., -10:]



        logprobs = logits.softmax(-1)
