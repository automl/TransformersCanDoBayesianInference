"""Positional encodings for `src/transformer.py`.

Copied from `positional_encodings.py` (only `NoPositionalEncoding`).

Interface shared by all encodings:
    __init__(d_model, max_len=...)
    forward(x: (seq_len, batch_size, d_model)) -> x with positional information added, same shape

NOTE: the PFN is meant to be invariant to the order of the training (context) points, so no
positional information is added. Passing `pos_encoder=None` to `BaseTransformerModel` is equivalent.
"""
from torch import nn


class NoPositionalEncoding(nn.Module):
    """Identity: no positional information is added. `d_model` and `max_len` are ignored."""

    def __init__(self, d_model=None, max_len=None):
        super().__init__()

    def forward(self, x):
        return x


