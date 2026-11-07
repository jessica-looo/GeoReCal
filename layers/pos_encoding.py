import torch
from torch import nn


def positional_encoding(q_len, d_model):
    W_pos = torch.empty((q_len, d_model))
    nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=True)
