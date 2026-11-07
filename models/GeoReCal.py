import torch
from torch import nn
import torch.nn.functional as F
from layers.GeoReCal_EncDec import SharedPatchPredictionHead
from layers.GeoReCal_EncDec import TSTEncoder, TowerEncoder, MixtureExpertsLayer
from layers.pos_encoding import positional_encoding
import pywt
import ptwt

class Model(nn.Module):
    """Multiscale reconstruction model; returns [batch, time, 1] predictions."""
    def __init__(self, configs):

        super().__init__()

        n_heads:int = 16
        head_dropout:float = 0.2
        d_ff:int = 256
        attn_dropout:float = 0.
        dropout1:float = 0.
        dropout2:float = 0.2
        self.freq_num:int = 4
        self.n_vars = 1
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.domain_len = configs.domain_len
        self.use_static_context = (not configs.is_relation_pretraining or
                                   configs.data in ('ngr_ms_crossvar', 'pwl_ms_crossvar'))
        self.meta_prefix_len = configs.meta_prefix_len

        self.projection_x = nn.Linear(self.seq_len, self.seq_len)

        self.patch_num = (max(self.seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        tgt_len = self.patch_len  + self.stride * (self.patch_num - 1)
        self.s_begin = self.seq_len - tgt_len

        self.W_pos = positional_encoding(self.patch_num * self.n_vars, self.d_model)

        self.dropout1 = nn.Dropout(dropout1)

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)

        self.encoder = TSTEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, attn_dropout=attn_dropout,
                                  dropout=dropout1,
                                  n_layers=self.e_layers)
        self.decoder = TowerEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, attn_dropout=attn_dropout,
                                dropout=dropout1,
                                n_layers=self.d_layers)

        self.wavelet_name = 'haar'
        self.decomposition_levels = 3

        for param in [self.W_pos]:
            param.requires_grad = False
        for module in [self.patch_embed, self.encoder, self.decoder]:
            for param in module.parameters():
                param.requires_grad = False

        self.domain_EnMoE = MixtureExpertsLayer(num_virtual_tokens=self.domain_len,
                            token_dim=self.d_model, encoder_hidden_size=self.d_model, n_layers=self.e_layers)
        self.domain_DeMoE = MixtureExpertsLayer(num_virtual_tokens=self.domain_len,
                            token_dim=self.d_model, encoder_hidden_size=self.d_model, n_layers=self.d_layers)
        self.domain_tokens = torch.arange(self.domain_len).long()
        self.dropout2 = nn.Dropout(dropout2)

        self.pretrain_heads = nn.ModuleList([
            SharedPatchPredictionHead(
                self.d_model, self.patch_len, self.stride, self.seq_len, head_dropout
            )
        ])

        self.static_emb_dim = getattr(configs, 'static_emb_dim', 8)

        self.static_dim = getattr(configs, 'static_dim', 4) # Context feature dimension.

        if self.use_static_context:
            self.static_encoder = nn.Linear(self.static_dim, self.static_emb_dim)
            self.static_meta_encoder = nn.Sequential(
                nn.Linear(self.static_emb_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model)
            )
            self.static_kv_proj_en = nn.Linear(
                self.d_model,
                self.e_layers * 2 * self.meta_prefix_len * self.d_model
            )

            self.static_kv_proj_de = nn.Linear(
                self.d_model,
                self.d_layers * 2 * self.meta_prefix_len * self.d_model
            )

    def mixture_of_experts(self, x, bs, encode=True):
        if encode:
            past_key_values = self.domain_EnMoE(x, self.domain_tokens.to(x.device))  # [freq_num * bs, n_vars * patch_num, e_layers * 2 * d_model]
            past_key_values = past_key_values.reshape(bs * self.freq_num, -1, self.e_layers, 2, self.d_model).permute(2, 3, 0, 1, 4)  # [e_layers, 2, freq_num * bs, n_vars * patch_num, d_model]
        else:
            x = x.permute(0, 1, 3, 2).reshape(bs, -1, self.d_model)
            past_key_values = self.domain_DeMoE(x, self.domain_tokens.to(x.device))   # [bs x patch_num * n_vars x 2 * d_layers * d_model]
            past_key_values = past_key_values.reshape(bs, -1, self.d_layers, 2, self.d_model).permute(2, 3, 0, 1, 4)
        past_key_values = self.dropout2(past_key_values)
        return past_key_values # [e_layers, 2, freq_num * bs, n_vars * patch_num, d_model]

    def decomposed_wavelet_learning(self, x):
        """Reconstruct Haar A3/D3/D2/D1 branches without coefficient filtering."""
        bs, seq_len, n_vars = x.shape
        x_in = x.permute(0, 2, 1).reshape(-1, 1, seq_len)
        wavelet = pywt.Wavelet(self.wavelet_name)
        coeffs = ptwt.wavedec(x_in, wavelet, level=self.decomposition_levels)
        branches = []
        for i in range(len(coeffs)):
            isolated = [c if i == j else torch.zeros_like(c) for j, c in enumerate(coeffs)]
            branch = ptwt.waverec(isolated, wavelet)[..., :seq_len]
            branch = F.pad(branch, (0, seq_len - branch.shape[-1]))
            branches.append(branch)
        return torch.stack(branches).view(self.freq_num, bs, n_vars, seq_len).permute(0, 1, 3, 2) # [freq_num, bs, seq_len, n_vars]

    def static_to_encoder_prefix(self, e_site, bs):
        """
        e_site: [bs, static_emb_dim]
        return: [e_layers, 2, bs*freq_num, meta_prefix_len, d_model]
        """
        z_meta = self.static_meta_encoder(e_site)

        kv = self.static_kv_proj_en(z_meta)

        kv = kv.view(bs, self.e_layers, 2, self.meta_prefix_len, self.d_model)

        kv = kv.permute(1, 2, 0, 3, 4).contiguous()

        # Wavelet tokens flatten [frequency, batch], so prefixes use the same order.
        kv = kv.unsqueeze(2)

        kv = kv.expand(-1, -1, self.freq_num, -1, -1, -1)

        kv = kv.reshape(self.e_layers, 2, bs * self.freq_num, self.meta_prefix_len, self.d_model)

        kv = self.dropout2(kv)
        return kv

    def static_to_decoder_prefix(self, e_site, bs):
        """
        e_site: [bs, static_emb_dim]
        return: [d_layers, 2, bs, meta_prefix_len, d_model]
        """
        z_meta = self.static_meta_encoder(e_site)   # [bs, d_model]

        kv = self.static_kv_proj_de(z_meta)         # [bs, d_layers * 2 * Lm * d_model]
        kv = kv.view(bs, self.d_layers, 2, self.meta_prefix_len, self.d_model)
        kv = kv.permute(1, 2, 0, 3, 4).contiguous() # [d_layers, 2, bs, Lm, d_model]

        kv = self.dropout2(kv)
        return kv

    def encode_series(self, x_enc, x_static=None):
        """Return pooled encoder features [bs, d_model] for relation pretraining."""
        if self.use_static_context and x_static is not None:
            e_site = self.static_encoder(x_static)
        else:
            e_site = None

        x = x_enc

        x = self.decomposed_wavelet_learning(x) # [freq_num, bs, seq_len, n_vars]
        x = x[:, :, self.s_begin:, :]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride) # [freq_num, bs, patch_num, n_vars, patch_len]
        freq_num, bs, patch_num, n_vars, patch_len = x.shape
        x = x.reshape(-1, patch_num, n_vars, patch_len) # [freq_num * bs, patch_num, n_vars, patch_len]
        x = self.patch_embed(x) # [freq_num * bs, patch_num, n_vars, d_model]

        x = x.transpose(1, 2) # [freq_num * bs, n_vars, patch_num, d_model]
        u = torch.reshape(x, (-1, n_vars * patch_num, self.d_model)) # [freq_num * bs, n_vars * patch_num, d_model]
        u = self.dropout1(u + self.W_pos)

        u_d = self.mixture_of_experts(u, bs=bs, encode=True) # [e_layers, 2, freq_num * bs, n_vars * patch_num, d_model]
        if e_site is not None:
            u_d_static = self.static_to_encoder_prefix(e_site, bs)
            u_d = torch.cat([u_d, u_d_static], dim=3) 

        x = self.encoder(u, u_d)
        x = x.reshape(-1, n_vars, patch_num, self.d_model)
        x = x.permute(0, 1, 3, 2)
        x = x.view(-1, bs, n_vars, self.d_model, patch_num).mean(dim=0)
        return x.mean(dim=(1, 3))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static=None):
        if self.use_static_context and x_static is not None:
            e_site = self.static_encoder(x_static)
        else:
            e_site = None

        x = x_enc

        x = self.decomposed_wavelet_learning(x)

        x = x[:, :, self.s_begin:, :]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [freq_num x bs x patch_num x n_vars x patch_len]
        freq_num, bs, patch_num, n_vars, patch_len = x.shape
        x = x.reshape(-1, patch_num, n_vars, patch_len)

        x = self.patch_embed(x)    # [bs * freq_num x patch_num x n_vars x d_model]

        x = x.transpose(1, 2)  # [bs * freq_num x n_vars x patch_num x d_model]
        u = torch.reshape(x, (-1, n_vars * patch_num, self.d_model))  # [bs' x n_vars * patch_num x d_model]
        u = self.dropout1(u + self.W_pos)  # [bs' x n_vars * patch_num x d_model]

        u_d = self.mixture_of_experts(u, bs=bs, encode=True)  # [e_layers x 2 x bs' x patch_num * n_vars x d_model]

        if e_site is not None:
            u_d_static = self.static_to_encoder_prefix(e_site, bs)
            u_d = torch.cat([u_d, u_d_static], dim=3)

        x = self.encoder(u, u_d)  # [bs * freq_num, n_vars * patch_num, d_model]
        x = x.reshape(-1, n_vars, patch_num, self.d_model)  # [bs' x n_vars x patch_num x d_model]
        x = x.permute(0, 1, 3, 2)  # [bs' x n_vars x d_model x patch_num]

        x = x.view(-1, bs, n_vars, self.d_model, patch_num).mean(dim=0)  # [bs x n_vars x d_model x patch_num]

        x_d = self.mixture_of_experts(x, bs=bs, encode=False)  # [d_layers x 2 x bs x patch_num * n_vars x d_model]

        if e_site is not None:
            x_d_static = self.static_to_decoder_prefix(e_site, bs)
            x_d = torch.cat([x_d, x_d_static], dim=3)

        x = self.decoder(x, x_d)  # [bs x n_vars x d_model x patch_num]

        return self.pretrain_heads[0](x)

    def get_maml_parameters(self):
        """Select trainable parameters for first-order MAML and freeze the rest."""
        for name, param in self.named_parameters():
            param.requires_grad = False

        maml_module_names = [
            'pretrain_heads',          # Reconstruction head.
            'domain_EnMoE.gate',       # Encoder routing.
            'domain_EnMoE.embedding',
            'domain_EnMoE.transform',
            'domain_DeMoE.gate',       # Decoder routing.
            'domain_DeMoE.embedding',
            'domain_DeMoE.transform',
            'static_encoder',          # Site context encoder.
            'static_meta_encoder',
            'static_kv_proj_en',
            'static_kv_proj_de',
            'patch_embed',             # Patch embedding.
            'W_pos',
        ]
        maml_module_names.extend(['encoder.', 'decoder.'])

        for name, param in self.named_parameters():
            for maml_module in maml_module_names:
                if maml_module in name:
                    param.requires_grad = True
                    break

        maml_params = {name: param for name, param in self.named_parameters() if param.requires_grad}

        return maml_params
