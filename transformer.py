import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn

from utils import SeqBN


class TransformerModel(nn.Module):
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers, dropout=0.0, y_encoder=None, pos_encoder=None, decoder=None, input_normalization=False):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.decoder = decoder(ninp, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(ninp) if input_normalization else None

        self.init_weights()

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz,sz) == 0
        mask[:,train_size:].zero_()
        mask |= torch.eye(sz) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 1.
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        assert single_eval_pos is not None, 'Single eval pos is required now.'
        fuse_x_y = not isinstance(src, tuple)
        assert not(fuse_x_y and single_eval_pos is not None), \
            'Don\'t use both fuxe_x_y and single_eval_pos (permutation equivariant setup) at the same time.'
        if src_mask is None:
            x_src = src if fuse_x_y else src[0]
            if single_eval_pos is None:
                src_mask = self.generate_square_subsequent_mask(len(x_src) if fuse_x_y else 2*len(x_src)).to(x_src.device)
            else:
                src_mask = self.generate_D_q_matrix(len(x_src), len(x_src)-single_eval_pos).to(x_src.device)
        if not fuse_x_y:
            x_src, y_src = src
            x_src = self.encoder(x_src)
            y_src = self.y_encoder(y_src.unsqueeze(-1))
            if single_eval_pos is None:
                src = torch.stack([x_src, y_src], 1).view(-1, *x_src.shape[1:])
            else:
                train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
                src = torch.cat([train_x, x_src[single_eval_pos:]], 0)
        else:
            src = self.encoder(src)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        if fuse_x_y:
            return output
        elif single_eval_pos is None:
            return output[0::2]
        else:
            return output[single_eval_pos:]
