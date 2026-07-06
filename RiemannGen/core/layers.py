"""Generic Perceiver-encoder / Transformer-decoder building blocks shared
by every backend's latent sequence model (atom's plain-SMILES model,
fragment's SAFE model)."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sdp(q, k, v, mask=None):
    # mask: bool tensor, True = masked out (-inf). Must already be broadcastable to
    # (B, heads, Tq, Tk), e.g. (1,1,Tq,Tk) for a causal mask or (B,1,1,Tk) for a
    # key-padding mask.
    attn = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if mask is not None:
        attn = attn.masked_fill(mask, float("-inf"))
    return torch.matmul(F.softmax(attn, dim=-1), v)


def _split_qkv(linear, x, nh, hd):
    B, T, _ = x.shape
    out = linear(x).reshape(B, T, 3, nh, hd).permute(2, 0, 3, 1, 4)
    return out[0], out[1], out[2]


def _split_kv(linear, x, nh, hd):
    B, T, _ = x.shape
    out = linear(x).reshape(B, T, 2, nh, hd).permute(2, 0, 3, 1, 4)
    return out[0], out[1]


def _heads(linear, x, nh, hd):
    B, T, _ = x.shape
    return linear(x).reshape(B, T, nh, hd).transpose(1, 2)


def _unheads(x):
    B, H, T, d = x.shape
    return x.transpose(1, 2).reshape(B, T, H * d)


class PerceiverCrossAttnLayer(nn.Module):
    def __init__(self, H: int, nh: int):
        super().__init__()
        hd = H // nh
        self.nh, self.hd = nh, hd
        self.ln_self   = nn.LayerNorm(H, eps=1e-5)
        self.self_qkv  = nn.Linear(H, 3 * H)
        self.self_out  = nn.Linear(H, H)
        self.ln_cross  = nn.LayerNorm(H, eps=1e-5)
        self.cross_q   = nn.Linear(H, H)
        self.cross_kv  = nn.Linear(H, 2 * H)
        self.cross_out = nn.Linear(H, H)
        self.ln_ffn = nn.LayerNorm(H, eps=1e-5)
        self.fc1    = nn.Linear(H, 4 * H)
        self.fc2    = nn.Linear(4 * H, H)

    def forward(self, latent: torch.Tensor, context: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r = latent
        h = self.ln_self(latent)
        q, k, v = _split_qkv(self.self_qkv, h, self.nh, self.hd)
        latent = self.self_out(_unheads(_sdp(q, k, v))) + r  # latent slots: no padding, no mask needed

        r = latent
        h = self.ln_cross(latent)
        q  = _heads(self.cross_q, h, self.nh, self.hd)
        k, v = _split_kv(self.cross_kv, context, self.nh, self.hd)
        latent = self.cross_out(_unheads(_sdp(q, k, v, mask=key_padding_mask))) + r

        r = latent
        h = self.ln_ffn(latent)
        latent = self.fc2(F.gelu(self.fc1(h))) + r
        return latent


class LatentSelfAttnLayer(nn.Module):
    def __init__(self, H: int, nh: int):
        super().__init__()
        hd = H // nh
        self.nh, self.hd = nh, hd
        self.ln_self  = nn.LayerNorm(H, eps=1e-5)
        self.self_qkv = nn.Linear(H, 3 * H)
        self.self_out = nn.Linear(H, H)
        self.ln_ffn = nn.LayerNorm(H, eps=1e-5)
        self.fc1    = nn.Linear(H, 4 * H)
        self.fc2    = nn.Linear(4 * H, H)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        r = latent
        h = self.ln_self(latent)
        q, k, v = _split_qkv(self.self_qkv, h, self.nh, self.hd)
        latent = self.self_out(_unheads(_sdp(q, k, v))) + r

        r = latent
        h = self.ln_ffn(latent)
        latent = self.fc2(F.gelu(self.fc1(h))) + r
        return latent


class DecoderLayer(nn.Module):
    def __init__(self, H: int, nh: int):
        super().__init__()
        hd = H // nh
        self.nh, self.hd = nh, hd
        self.ln_self   = nn.LayerNorm(H, eps=1e-5)
        self.self_qkv  = nn.Linear(H, 3 * H)
        self.self_out  = nn.Linear(H, H)
        self.ln_cross  = nn.LayerNorm(H, eps=1e-5)
        self.cross_q   = nn.Linear(H, H)
        self.cross_kv  = nn.Linear(H, 2 * H)
        self.cross_out = nn.Linear(H, H)
        self.ln_ffn = nn.LayerNorm(H, eps=1e-5)
        self.fc1    = nn.Linear(H, 4 * H)
        self.fc2    = nn.Linear(4 * H, H)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r = x
        h = self.ln_self(x)
        q, k, v = _split_qkv(self.self_qkv, h, self.nh, self.hd)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)
        T, Tc = q.shape[2], k.shape[2]
        mask = None if kv_cache is not None else \
            torch.triu(torch.ones(T, Tc, device=x.device), 1).bool().unsqueeze(0).unsqueeze(0)
        x = self.self_out(_unheads(_sdp(q, k, v, mask))) + r

        r = x
        h = self.ln_cross(x)
        cq = _heads(self.cross_q, h, self.nh, self.hd)
        ck, cv = _split_kv(self.cross_kv, enc_out, self.nh, self.hd)
        x = self.cross_out(_unheads(_sdp(cq, ck, cv))) + r

        r = x
        h = self.ln_ffn(x)
        x = self.fc2(F.gelu(self.fc1(h))) + r
        return x, new_kv
