"""PFN Transformer model.

Copied from `transformer.py`, without the legacy `fuse_x_y` option: the input is always an
`(x, y)` tuple and `single_eval_pos` is always required. Removing that option also removed
the code paths for `single_eval_pos=None` (causal mask, interleaved x/y tokens). They could
not be reached anyway.
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .utils import SeqBN


class TransformerModel(nn.Module):
    """PFN predicting a Gaussian `N(mean, var)` for every query point.

    The decoder produces two raw values per query point: the mean and an unconstrained value that is
    mapped to the variance via `softplus(raw) + min_var` (smooth, positive, ~0.69 at init).

    :param decoder: optional factory `decoder(ninp, nhid, n_out)` with `n_out = 2`; default is an MLP
    :param min_var: lower bound added to the predicted variance for numerical stability

    Attribute `criterion`: `nn.GaussianNLLLoss(reduction='none', full=True)`, called as
    `criterion(mean, target, var=var)`.
    """
    n_out = 2  # mean and (raw) variance

    def __init__(
            self,
            encoder, ninp, nhead, nhid, nlayers,
            dropout=0.0,
            y_encoder=None, pos_encoder=None, decoder=None,
            input_normalization=False, min_var=1e-6
    ):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.min_var = min_var
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.decoder = (
            decoder(ninp, nhid, self.n_out) if decoder is not None
            else nn.Sequential(nn.Linear(ninp, nhid),
                               nn.GELU(),
                               nn.Linear(nhid, self.n_out))
        )
        self.input_ln = SeqBN(ninp) if input_normalization else None
        # Loss matching the predicted Gaussian; per-element losses, incl. the constant term.
        self.criterion = nn.GaussianNLLLoss(reduction='none', full=True)

        self.init_weights()

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        """Attention mask: every position attends to all training (context) positions and to itself;
        query positions do not attend to each other. 0 = allowed, -inf = masked."""
        train_size = sz - query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        # Zero the residual-branch outputs so every encoder layer starts as the identity.
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        """
        :param src: tuple `(x, y)` with x: (seq_len, batch_size, num_features), y: (seq_len, batch_size)
        :param src_mask: optional attention mask; defaults to `generate_D_q_matrix`
        :param single_eval_pos: number of training (context) positions; positions from here on are queries
        :return: `(mean, var)` of the predictive distribution at the query positions,
            each of shape (seq_len - single_eval_pos, batch_size)
        """
        assert single_eval_pos is not None, 'single_eval_pos is required.'
        assert isinstance(src, tuple), 'src must be a tuple (x, y).'
        x_src, y_src = src

        if src_mask is None:
            src_mask = self.generate_D_q_matrix(len(x_src), len(x_src) - single_eval_pos).to(x_src.device)

        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1))
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output[single_eval_pos:])
        mean = output[..., 0]
        var = nn.functional.softplus(output[..., 1]) + self.min_var
        return mean, var
