"""Relation pretraining through the shared encoder: r = MLP([zx, zy, zx * zy])."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GeoReCal import Model as GeoReCalModel


class Model(nn.Module):
    """
    Relation self-supervised wrapper around GeoReCal encoder.
    """

    def __init__(self, configs):
        super().__init__()
        self.backbone = GeoReCalModel(configs)
        d_model = configs.d_model
        self.relation_projector = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._set_pretrain_trainable()

    def _set_pretrain_trainable(self):
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False

        trainable_prefixes = (
            'W_pos',
            'projection_x',
            'patch_embed',
            'encoder',
            'domain_EnMoE',
            'static_encoder',
            'static_meta_encoder',
            'static_kv_proj_en',
        )
        for name, param in self.backbone.named_parameters():
            if name.startswith(trainable_prefixes):
                param.requires_grad = True        

    def forward(self, x, y, static=None):
        zx = self.backbone.encode_series(x, x_static=static)
        zy = self.backbone.encode_series(y, x_static=static)
        rel = torch.cat([zx, zy, zx * zy], dim=-1)
        rel = self.relation_projector(rel)
        return F.normalize(zx, dim=-1), F.normalize(zy, dim=-1), F.normalize(rel, dim=-1)
