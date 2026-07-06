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
from typing import Dict, Optional

import torch
import torch.nn as nn

# Registry of known, already-trained heads shipped in the atom research
# repo. `head="tanimoto"` (or any other key here) resolves through this; an
# unrecognized name is treated as a literal checkpoint path instead.
KNOWN_HEADS: Dict[str, str] = {
    "tanimoto": "atom/checkpoints/metric_head/metric_head_infonce_1M.pt",
    "potency_egfr": "atom/checkpoints/metric_head/potency_metric_head_egfr.pt",
}


class MetricHead:
    def __init__(self, proj: nn.Module, input_dim: int, proj_dim: int, metadata: Optional[dict] = None):
        self.proj = proj
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.metadata = metadata or {}

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., input_dim). Returns (..., proj_dim)."""
        with torch.no_grad():
            return self.proj(z)

    def distance(self, z1: torch.Tensor, z2: torch.Tensor) -> float:
        p1, p2 = self.project(z1), self.project(z2)
        return torch.linalg.vector_norm(p1 - p2).item()


def load_metric_head(checkpoint_path: str | Path) -> MetricHead:
    """Generic loader: infers input_dim/proj_dim from the checkpoint's own
    saved weight shapes rather than hardcoding specific sizes, so this
    works for any base z size without new code. Handles both checkpoint
    formats found in the repo: a bare state_dict, and a dict wrapping
    state_dict + training metadata."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        state_dict = ckpt
        metadata = {}

    hidden_dim, input_dim = state_dict["0.weight"].shape
    proj_dim, _ = state_dict["2.weight"].shape

    proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim))
    proj.load_state_dict(state_dict)
    proj.eval()
    return MetricHead(proj, input_dim, proj_dim, metadata)


def resolve_head_path(name_or_path: str, repo_root: Optional[Path] = None) -> str:
    """Resolves a --head argument: a KNOWN_HEADS name, or a literal path."""
    if name_or_path in KNOWN_HEADS:
        rel = KNOWN_HEADS[name_or_path]
        return str((repo_root / rel)) if repo_root else rel
    return name_or_path
