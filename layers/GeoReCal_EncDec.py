import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

from layers.basics import RMSNorm
from layers.attention import MultiheadAttention

class SharedPatchPredictionHead(nn.Module):
    """Decode each patch with shared weights, then average overlapping years."""

    def __init__(self, d_model, patch_len, stride, forecast_len, dropout=0.0):
        super().__init__()
        assert 0 < stride <= patch_len
        self.stride = stride
        self.forecast_len = forecast_len
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        patches = self.linear(self.dropout(x.transpose(-1, -2)))
        batch, channels, patch_num, patch_len = patches.shape
        length = (patch_num - 1) * self.stride + patch_len
        assert length == self.forecast_len
        output = patches.new_zeros(batch, channels, length)
        count = patches.new_zeros(length)
        for i in range(patch_num):
            start = i * self.stride
            output[:, :, start:start + patch_len] += patches[:, :, i, :]
            count[start:start + patch_len] += 1
        return (output / count).transpose(1, 2)

class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, attn_dropout=0.,
                 dropout=0., n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff,
                                                      attn_dropout=attn_dropout, dropout=dropout)
                                    for i in range(n_layers)])

    def forward(self, src:Tensor, prefix_d: torch.Tensor = None):
        """
        src: tensor [bs x seq_len x d_model]
        prefix_d: tensor [n_layers x 2 x bs x domain_len x d_model]
        """
        output = src
        scores = None
        for i, layer in enumerate(self.layers):
            prefix = prefix_d[i] if prefix_d is not None else None
            output, scores = layer(output, prefix, prev=scores)
        return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256,
                 attn_dropout=0, dropout=0.):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                            proj_dropout=dropout)

        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = RMSNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.SiLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))

        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = RMSNorm(d_model)


    def forward(self, src:Tensor, prefix_d: torch.Tensor = None, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x seq_len x d_model]
        prefix_d: tensor [2 x bs x domain_len x d_model]
        """

        src = self.norm_attn(src)
        if prefix_d is None:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            K = torch.cat([prefix_d[0], src], dim=1)
            V = torch.cat([prefix_d[1], src], dim=1)
            src2, attn, scores = self.self_attn(src, K, V, prev)
        src = src + self.dropout_attn(src2)

        src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)

        return src, scores

class TowerEncoder(nn.Module):
    """
    x: tensor [bs x n_vars x d_model x patch_num]
    prefix_d: tensor [n_layers x 2 x bs x domain_len x d_model]
    """
    def __init__(self, d_model, n_heads, d_ff=None,
                        attn_dropout=0., dropout=0.,
                        n_layers=1):

        super().__init__()
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                                   n_layers=n_layers)

    def forward(self, x, prefix_d: torch.Tensor = None):
        bs, n_vars, d_model, patch_num = x.shape
        x = x.permute(0,1,3,2)
        x = x.reshape(bs, n_vars * patch_num, d_model)
        x = self.encoder(x, prefix_d)
        x = x.reshape(bs, n_vars, patch_num, d_model)
        x = x.permute(0,1,3,2)
        return x

class MixtureExpertsLayer(torch.nn.Module):
    """Generate layer-wise attention prefixes from routed expert embeddings.
    Inputs: x [bs, seq_len, token_dim], prefix [num_virtual_tokens].
    Output: [bs, seq_len, n_layers * 2 * token_dim].
    """
    def __init__(self, num_virtual_tokens, token_dim, encoder_hidden_size, n_layers):
        super().__init__()
        self.gate = nn.Linear(token_dim, num_virtual_tokens)
        self.embedding = nn.Embedding(num_virtual_tokens, token_dim)
        self.transform = nn.Sequential(
            nn.Linear(token_dim, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(
                encoder_hidden_size,
                n_layers * 2 * token_dim,
            ),
        )

    def forward(self, x: torch.Tensor, prefix: torch.Tensor):
        router_logits = self.gate(x)  # [freq_num * bs, n_vars * patch_num, token_dim] -> [freq_num * bs, n_vars * patch_num, num_virtual_tokens]
        routing_weights = F.softmax(router_logits, dim=2, dtype=torch.float) # s_i,p 

        prefix_tokens = self.embedding(prefix) # [num_virtual_tokens, token_dim]

        prefix_tokens = torch.einsum('...e,ed->...d', routing_weights, prefix_tokens) # [freq_num * bs, n_vars * patch_num, token_dim]
        past_key_values = self.transform(prefix_tokens) # [freq_num * bs, n_vars * patch_num, n_layers * 2 * token_dim]
        return past_key_values
