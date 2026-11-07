import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dropout=0., proj_dropout=0.):
        """Multi-head attention.
        Q: [bs, q_len, d_model]; K/V: [bs, kv_len, d_model].
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout)

        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Project and split into attention heads.
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s: [bs, n_heads, q_len, d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s: [bs, n_heads, d_k, kv_len]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s: [bs, n_heads, kv_len, d_v]

        output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs, n_heads, q_len, d_v]; weights/scores: [bs, n_heads, q_len, kv_len]

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # [bs, q_len, n_heads * d_v]
        output = self.to_out(output)

        return output, attn_weights, attn_scores

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention (Vaswani et al., 2017), with residual scores
    (He et al., 2020). The training stage controls scale gradients.
    """

    def __init__(self, d_model, n_heads, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        """q: [bs, n_heads, q_len, d_k]; k: [bs, n_heads, d_k, kv_len].
        v: [bs, n_heads, kv_len, d_v]; prev: [bs, n_heads, q_len, kv_len].
        key_padding_mask: [bs, kv_len]; attn_mask broadcasts over attention scores.
        Returns output [bs, n_heads, q_len, d_v] and weights [bs, n_heads, q_len, kv_len];
        residual attention also returns scores with the same shape as weights.
        """

        attn_scores = torch.matmul(q, k) * self.scale      # [bs, n_heads, q_len, kv_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # Mask broadcastable to [bs, n_heads, q_len, kv_len].
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # [bs, kv_len]
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)                 # [bs, n_heads, q_len, kv_len]
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)                        # [bs, n_heads, q_len, d_v]

        return output, attn_weights, attn_scores
