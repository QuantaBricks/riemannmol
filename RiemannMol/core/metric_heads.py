"""Pluggable metric heads: small supervised projection heads trained on top
of a frozen base z (Linear(input_dim,256)->ReLU->Linear(256,proj_dim)) so
that distance(head(z_a), head(z_b)) approximates a real domain notion of
similarity the base z alone doesn't capture -- e.g. Tanimoto structural
similarity, or per-target potency similarity. This module is just the
generic "attach a trained head to a base z" mechanism, not training code.

A metric head only helps for the property it was actually trained on (a
Tanimoto-trained head does not transfer to potency, and vice versa) --
attaching the wrong head for your task will not error, it will just
silently give an uninformative distance.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .hf_download import resolve_path

# Registry of known, already-trained heads shipped in the atom research
# repo. `head="tanimoto"` (or any other key here) resolves through this; an
# unrecognized name is treated as a literal checkpoint path instead. Each
# entry is (local repo-relative path, HF Hub filename in
# QuantaBricks/riemannmol) -- resolve_head_path falls back to downloading
# the HF filename when the local path doesn't exist (e.g. a pip install
# with no full repo checkout).
KNOWN_HEADS: Dict[str, Tuple[str, str]] = {
    "tanimoto": ("atom/checkpoints/metric_head/metric_head_infonce_1M.pt",
                 "atom/metric_head_infonce_1M.pt"),
    "potency_egfr": ("atom/checkpoints/metric_head/potency_metric_head_egfr.pt",
                      "atom/potency_metric_head_egfr.pt"),
}


class CouplingLayer(nn.Module):
    """One RealNVP-style block: transforms one half of z conditioned on the
    untouched other half, so the whole stack is exactly invertible even
    though the scale/translate network (self.st) is non-linear."""

    def __init__(self, dim, hidden=192, swap=False):
        super().__init__()
        self.half = dim // 2
        self.swap = swap
        self.st = nn.Sequential(nn.Linear(self.half, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, self.half * 2))

    def forward(self, z):
        if self.swap:
            z_b, z_a = z[..., : self.half], z[..., self.half:]
        else:
            z_a, z_b = z[..., : self.half], z[..., self.half:]
        s, t = self.st(z_a).chunk(2, dim=-1)
        s = torch.tanh(s)
        z_b_new = z_b * torch.exp(s) + t
        return torch.cat([z_b_new, z_a], dim=-1) if self.swap else torch.cat([z_a, z_b_new], dim=-1)

    def invert(self, z):
        if self.swap:
            z_b_new, z_a = z[..., : self.half], z[..., self.half:]
        else:
            z_a, z_b_new = z[..., : self.half], z[..., self.half:]
        s, t = self.st(z_a).chunk(2, dim=-1)
        s = torch.tanh(s)
        z_b = (z_b_new - t) * torch.exp(-s)
        return torch.cat([z_b, z_a], dim=-1) if self.swap else torch.cat([z_a, z_b], dim=-1)


class CouplingStack(nn.Module):
    """Stack of CouplingLayers, exactly invertible by composition. Output
    dim equals input dim (no bottleneck) -- required for invertibility."""

    def __init__(self, dim=384, n_layers=6, hidden=192):
        super().__init__()
        self.layers = nn.ModuleList([CouplingLayer(dim, hidden, swap=(i % 2 == 1)) for i in range(n_layers)])

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z

    def invert(self, z):
        for layer in reversed(self.layers):
            z = layer.invert(z)
        return z


class MetricHead:
    def __init__(self, proj: nn.Module, input_dim: int, proj_dim: int, metadata: Optional[dict] = None):
        self.proj = proj
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.metadata = metadata or {}
        self.invertible = hasattr(proj, "invert")

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., input_dim). Returns (..., proj_dim)."""
        with torch.no_grad():
            return self.proj(z)

    def invert(self, projected: torch.Tensor) -> torch.Tensor:
        """Recovers the original z from a projected value. Only available
        when the underlying architecture is exactly invertible (coupling
        layers) -- raises otherwise."""
        if not self.invertible:
            raise NotImplementedError(
                "this head's architecture is not invertible (e.g. it reduces "
                "dimensionality or uses a non-invertible non-linearity); "
                "use the 'tanimoto' coupling-layer head for invert()"
            )
        with torch.no_grad():
            return self.proj.invert(projected)

    def distance(self, z1: torch.Tensor, z2: torch.Tensor) -> float:
        p1, p2 = self.project(z1), self.project(z2)
        return torch.linalg.vector_norm(p1 - p2).item()


def load_metric_head(checkpoint_path: str | Path) -> MetricHead:
    """Generic loader: infers architecture and shapes from the checkpoint's
    own saved weights rather than hardcoding them, so this works for any
    base z size without new code. Handles both checkpoint formats found in
    the repo: a bare state_dict, and a dict wrapping state_dict + training
    metadata (with an explicit "architecture" key for non-default heads)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        state_dict = ckpt
        metadata = {}

    if metadata.get("architecture") == "coupling":
        dim = metadata.get("dim", 384)
        n_layers = metadata.get("n_layers", 6)
        hidden = metadata.get("hidden", 192)
        proj = CouplingStack(dim=dim, n_layers=n_layers, hidden=hidden)
        proj.load_state_dict(state_dict)
        proj.eval()
        return MetricHead(proj, dim, dim, metadata)

    hidden_dim, input_dim = state_dict["0.weight"].shape
    proj_dim, _ = state_dict["2.weight"].shape

    proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim))
    proj.load_state_dict(state_dict)
    proj.eval()
    return MetricHead(proj, input_dim, proj_dim, metadata)


def resolve_head_path(name_or_path: str, repo_root: Optional[Path] = None) -> str:
    """Resolves a --head argument: a KNOWN_HEADS name (downloading from the
    HF Hub if the local repo-relative path doesn't exist), or a literal
    path used as-is."""
    if name_or_path in KNOWN_HEADS:
        rel, hf_filename = KNOWN_HEADS[name_or_path]
        local_path = (repo_root / rel) if repo_root else Path(rel)
        return resolve_path(local_path, hf_filename)
    return name_or_path
